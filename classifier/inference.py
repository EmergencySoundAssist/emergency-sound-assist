"""
① 소리 분류 — ONNX 추론 (cnn_attn_full, ViT-CNN-Attention deploy 브랜치 산출).

모델: classifier/models/cnn_attn_full_s42.onnx (+ .onnx.data)
  - 입력  : mel (1,1,64,216) float
  - 출력  : logits (1,3) = {siren, horn, noise}  (학습 클래스 순서)

전처리는 학습 레포 dataset.py의 logmel을 **그대로 복제** → 학습/추론 skew 차단.
  - 모델 입력 레이트 22.05kHz (런타임 16kHz → 리샘플)
  - 5초 윈도우(216 프레임) → 매 청크 누적하는 **상태 보유**(롤링 버퍼)
  - 윈도우별 정규화 (x-μ)/σ
출력 매핑: noise → normal_traffic (core.types 약속).
차종(선택): siren 이면 subtype_cnn_attn ONNX로 구급/경찰/소방을 추가 추론한다
           (검출과 같은 멜 윈도우 재사용). 모델 파일이 없으면 자동 생략,
           차종 확신<0.6 이면 '긴급차량'(UNKNOWN)으로 일반화.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.types import AudioChunk, ClassResult, SoundClass, SirenSubtype

# ── 학습(dataset.py)과 반드시 동일해야 하는 상수 ──────────────────────
SR_MODEL = 22050
N_MELS, N_FFT, HOP = 64, 1024, 512
N_FRAMES = 216                      # 5초 윈도우
WIN_S = 5.0
LOG_EPS = 1e-6
PAD_VAL = float(np.log(LOG_EPS))    # 짧은 꼬리 무음 패딩값
BUF_SAMPLES = int(WIN_S * SR_MODEL)

CLASSES = ("siren", "horn", "noise")          # ONNX 출력 순서 (학습과 동일)
_TO_SOUNDCLASS = {                             # 학습 클래스 → core.types
    "siren": SoundClass.SIREN,
    "horn": SoundClass.HORN,
    "noise": SoundClass.NORMAL_TRAFFIC,
}
_MODELS = Path(__file__).resolve().parent / "models"
# 실차 파인튜닝판(_early2)이 기본이다. 런타임 경로 실측(워밍업 틱 제외, 실차 17클립):
#   지연 중앙 5.0초 → 2.0초 · 네거티브 오발화 14.7% → 9.0%(클립 45/49 → 16/49)
#   사이렌 놓침 0/17 유지. 지연과 오경보가 함께 줄었다 (docs/collect/latency.md).
# 구모델로 즉시 되돌리려면 ESA_DETECT_MODEL=cnn_attn_full_s42.onnx 로 실행한다.
_MODEL_PATH = _MODELS / os.environ.get("ESA_DETECT_MODEL", "cnn_attn_full_s42_early2.onnx")

# ── 차종(사이렌 세분화) — ViT subtype_clf 모델 (선택) ────────────────
# 검출과 입력(64×216 멜)·정규화가 동일 → 같은 윈도우를 그대로 재사용한다.
# 모델 파일이 없으면 차종은 생략하고 기존처럼 'siren'으로만 출력(graceful).
SUBS = ("ambulance", "police", "fire")        # subtype_clf.SUBS=[구급차,경찰차,소방차] 순서
_TO_SUBTYPE = {
    "ambulance": SirenSubtype.AMBULANCE,
    "police": SirenSubtype.POLICE,
    "fire": SirenSubtype.FIRE,
}
_SUBTYPE_PATH = Path(__file__).resolve().parent / "models" / "subtype_cnn_attn_yt_s42.onnx"   # yt=실채널 파인튜닝판
SUBTYPE_CONF = 0.6        # 미만이면 '긴급차량'(UNKNOWN)으로 일반화 (경찰↔구급 혼동 회피)

# ── 속도+움직임 증거 (speed_neural_dir — 정지/접근/멀어짐 헤드) ────────
# 출력 3개: speed(진단용)·f0(미사용)·dir(0=정지/1=접근/2=멀어짐).
# 최종 이동 판단은 dir 확률을 음량·직접 도플러와 조건부로 융합한다.
_SPEED_PATH = Path(__file__).resolve().parent / "models" / "speed_neural_dir.onnx"

# ── 2초 예비검출 (석우 이중 창 — 같은 가중치, 창만 87프레임) ────────────
# 5초 확정보다 먼저 우는 "빠른 귀" (PRE 예비경보 ~1.8s). 없으면 이중 창 생략.
_FAST_PATH = _MODELS / (
    "cnn_attn_full_s42_early2_87f.onnx"
    if _MODEL_PATH.name.startswith("cnn_attn_full_s42_early")
    else "cnn_attn_full_s42_87f.onnx")     # 예비 창은 확정 창과 같은 가중치를 쓴다
FAST_FRAMES = 87


@dataclass(frozen=True)
class SpeedEvidence:
    """``speed_neural_dir``의 원시 증거.

    ``speed_kmh``는 공개 데이터 교차검증에서 차량 속도로 검증되지 않았으므로 진단 로그에만
    남긴다. 융합에는 정지/접근/멀어짐 순서의 ``dir_probabilities``만 사용한다.
    """

    speed_kmh: float
    dir_probabilities: tuple[float, float, float]

    @property
    def direction_index(self) -> int:
        return int(np.argmax(self.dir_probabilities))

    @property
    def confidence(self) -> float:
        return float(max(self.dir_probabilities))


# ── 전처리 (dataset.py logmel 복제) ──────────────────────────────────
def _mel_fb(sr=SR_MODEL, n_fft=N_FFT, n_mels=N_MELS) -> np.ndarray:
    """HTK mel 삼각 필터뱅크 (64, 513)."""
    mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    pts = 700.0 * (10.0 ** (np.linspace(mel(0.0), mel(sr / 2), n_mels + 2) / 2595.0) - 1.0)
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), np.float32)
    for i in range(n_mels):
        l, c, r = bins[i], bins[i + 1], bins[i + 2]
        c = max(c, l + 1)
        r = max(r, c + 1)
        fb[i, l:c] = (np.arange(l, c) - l) / (c - l)
        fb[i, c:r] = (r - np.arange(c, r)) / (r - c)
    return fb


_FB = _mel_fb()
_WINDOW = np.hanning(N_FFT + 1)[:-1].astype(np.float32)   # periodic hann


def _logmel(y: np.ndarray) -> np.ndarray:
    """(64, T) 로그멜 (dataset.py와 동일)."""
    y = np.pad(y, N_FFT // 2, mode="reflect")
    n = 1 + (len(y) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
    spec = np.abs(np.fft.rfft(y[idx] * _WINDOW, axis=1)) ** 2
    return np.log(spec @ _FB.T + LOG_EPS).T.astype(np.float32)


def _resample(y: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return y.astype(np.float32)
    from scipy.signal import resample_poly      # 지연 import
    g = math.gcd(int(src), int(dst))
    return resample_poly(y, dst // g, src // g).astype(np.float32)


def _mono(samples: np.ndarray) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float32)
    return x.mean(axis=1) if x.ndim > 1 else x


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def _norm_win(m: np.ndarray) -> np.ndarray:
    """윈도우별 정규화 (학습과 동일) — 확정 5초 창과 예비 2초 창을 각자 정규화."""
    return ((m - m.mean()) / (m.std() + 1e-5)).astype(np.float32)


# ── 추론기 (상태 보유: 5초 롤링 버퍼) ────────────────────────────────
class _OnnxClassifier:
    def __init__(self) -> None:
        self._sess = None
        self._in = None
        self._buf = np.zeros(0, dtype=np.float32)
        self._sub_sess = None
        self._sub_in = None
        self._sub_unavailable = False        # 차종 모델 없음 확인 후 재시도 방지
        self._spd_sess = None
        self._spd_in = None
        self._spd_unavailable = False        # 속도 모델 없음 확인 후 재시도 방지
        self._fast_sess = None
        self._fast_in = None
        self._fast_unavailable = False       # 예비검출(87f) 없음 확인 후 재시도 방지
        self._last_x = None                  # analyze()의 정규화 5초 창 — 차종/속도가 재사용

    def _session(self):
        if self._sess is None:
            import onnxruntime as ort            # 지연 import
            if not _MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"ONNX 모델이 없습니다: {_MODEL_PATH}\n"
                    "deploy 브랜치에서 cnn_attn_full_s42.onnx(+.data)를 가져오세요."
                )
            prefer = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            providers = [p for p in prefer if p in ort.get_available_providers()]
            self._sess = ort.InferenceSession(str(_MODEL_PATH), providers=providers)
            self._in = self._sess.get_inputs()[0].name
        return self._sess

    def _subtype_session(self):
        """차종 ONNX 세션 (lazy). 파일이 없으면 None — 차종 없이 동작한다."""
        if self._sub_sess is None and not self._sub_unavailable:
            if not _SUBTYPE_PATH.exists():
                self._sub_unavailable = True
                print(f"[classifier] 차종 모델 없음({_SUBTYPE_PATH.name}) → 차종 생략, 'siren'으로만 출력")
                return None
            import onnxruntime as ort            # 지연 import
            prefer = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            providers = [p for p in prefer if p in ort.get_available_providers()]
            self._sub_sess = ort.InferenceSession(str(_SUBTYPE_PATH), providers=providers)
            self._sub_in = self._sub_sess.get_inputs()[0].name
        return self._sub_sess

    def _infer_subtype(self, x: np.ndarray):
        """정규화된 멜 (1,1,64,216) → (SirenSubtype, conf) | (None, None).
        검출과 동일 입력이라 같은 윈도우 x를 그대로 넣는다. 사이렌일 때만 호출."""
        sess = self._subtype_session()
        if sess is None:
            return None, None
        logits = sess.run(None, {self._sub_in: x})[0][0]
        prob = _softmax(logits)
        j = int(prob.argmax())
        p = float(prob[j])
        sub = _TO_SUBTYPE[SUBS[j]] if p >= SUBTYPE_CONF else SirenSubtype.UNKNOWN
        return sub, p

    def _speed_session(self):
        """속도 ONNX 세션 (lazy). 파일이 없으면 None — 속도 없이 동작한다."""
        if self._spd_sess is None and not self._spd_unavailable:
            if not _SPEED_PATH.exists():
                self._spd_unavailable = True
                print(f"[classifier] 속도 모델 없음({_SPEED_PATH.name}) → 속도 단계 생략")
                return None
            import onnxruntime as ort            # 지연 import
            prefer = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            providers = [p for p in prefer if p in ort.get_available_providers()]
            self._spd_sess = ort.InferenceSession(str(_SPEED_PATH), providers=providers)
            self._spd_in = self._spd_sess.get_inputs()[0].name
        return self._spd_sess

    def _fast_session(self):
        """2초 예비검출(87f) 세션 (lazy). 파일 없으면 None — 이중 창 생략(graceful)."""
        if self._fast_sess is None and not self._fast_unavailable:
            if not _FAST_PATH.exists():
                self._fast_unavailable = True
                print(f"[classifier] 예비검출 모델 없음({_FAST_PATH.name}) → 이중 창 생략")
                return None
            import onnxruntime as ort            # 지연 import
            prefer = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            providers = [p for p in prefer if p in ort.get_available_providers()]
            self._fast_sess = ort.InferenceSession(str(_FAST_PATH), providers=providers)
            self._fast_in = self._fast_sess.get_inputs()[0].name
        return self._fast_sess

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)
        self._last_x = None

    # ── 하이브리드 API — 마진 판정 + 창 재사용 (석우 infer_trt.step 방식) ──
    def analyze(self, chunk: AudioChunk):
        """청크 누적 → 멜 1회 → 검출 로짓 **마진**(5초 확정 + 2초 예비).

        softmax(1.000 포화) 대신 마진 z[cls]-max(나머지)로 게이트 판정(alert.Gate 입력).
        정규화된 5초 창을 보관(self._last_x) → subtype_probs()/speed_evidence()가 재사용(멜 1회).
        반환: dict(label, conf, m_siren, m_horn, m_fast|None) — 버퍼 부족(워밍업)이면 None.
        """
        y = _resample(_mono(chunk.samples), chunk.sample_rate, SR_MODEL)
        self._buf = np.concatenate([self._buf, y])[-BUF_SAMPLES:]
        if self._buf.size < N_FFT:               # 아직 너무 짧음
            return None

        m = _logmel(self._buf)[:, -N_FRAMES:]    # 최근 5초 윈도우 (미정규화)
        if m.shape[1] < N_FRAMES:                # 버퍼가 5초 미만이면 끝을 무음 패딩
            m = np.pad(m, ((0, 0), (0, N_FRAMES - m.shape[1])), constant_values=PAD_VAL)
        x = np.ascontiguousarray(_norm_win(m)[None, None], dtype=np.float32)   # (1,1,64,216)

        z = self._session().run(None, {self._in: x})[0][0]
        m_siren = float(z[0] - max(z[1], z[2]))
        m_horn = float(z[1] - max(z[0], z[2]))
        prob = _softmax(z)
        i = int(prob.argmax())

        m_fast = None
        fs = self._fast_session()
        if fs is not None:                        # 예비: 같은 멜의 끝 2초만, 별도 정규화
            xf = np.ascontiguousarray(_norm_win(m[:, -FAST_FRAMES:])[None, None], dtype=np.float32)
            zf = fs.run(None, {self._fast_in: xf})[0][0]
            m_fast = float(zf[0] - max(zf[1], zf[2]))

        self._last_x = x
        return {"label": _TO_SOUNDCLASS[CLASSES[i]], "conf": float(prob[i]),
                "m_siren": m_siren, "m_horn": m_horn, "m_fast": m_fast}

    def subtype_probs(self):
        """마지막 analyze 창 → 차종 softmax 확률[3] (yt 파인튜닝판). 모델 없으면 None.
        투표(SubtypeVote)는 pipeline 이 담당 — 여기선 확률만."""
        sess = self._subtype_session()
        if sess is None or self._last_x is None:
            return None
        logits = sess.run(None, {self._sub_in: self._last_x})[0][0]
        return _softmax(logits)

    def speed_ready(self) -> bool:
        """속도/방향 모델에 무음 패딩이 없는 실제 5초 창이 준비됐는지 반환."""
        return self._buf.size >= BUF_SAMPLES and self._last_x is not None

    def speed_evidence(self):
        """마지막 완전한 5초 창 → :class:`SpeedEvidence` | None.

        5초 전 실행하면 오른쪽 무음 패딩을 차량 이동으로 오인하는 현상이 확인되어,
        검출 모델과 달리 이 모델은 실제 오디오가 5초 쌓이기 전에는 실행하지 않는다.
        """
        if not self.speed_ready():
            return None
        sess = self._speed_session()
        if sess is None:
            return None
        outs = sess.run(None, {self._spd_in: self._last_x})
        speed = None
        direction = None
        for value in outs:                       # 출력 이름이 바뀌어도 크기로 식별
            value = np.asarray(value)
            if value.size == 1:
                speed = max(0.0, float(value.reshape(-1)[0]))
            elif value.size == 3:
                direction = _softmax(value.reshape(-1))
        if speed is None or direction is None:
            return None
        return SpeedEvidence(
            speed_kmh=speed,
            dir_probabilities=tuple(float(p) for p in direction),
        )

    def infer(self, chunk: AudioChunk) -> ClassResult:
        """분류 단독 API. 사이렌이면 동일 5초 창에서 차종도 함께 반환한다."""
        res = self.analyze(chunk)
        if res is None:
            return ClassResult.from_label(SoundClass.NORMAL_TRAFFIC, 0.0)
        subtype = subtype_confidence = None
        if res["label"] is SoundClass.SIREN and self._last_x is not None:
            subtype, subtype_confidence = self._infer_subtype(self._last_x)
        return ClassResult.from_label(
            res["label"],
            res["conf"],
            subtype=subtype,
            subtype_confidence=subtype_confidence,
        )


_CLF = _OnnxClassifier()


def infer(chunk: AudioChunk) -> ClassResult:
    """AudioChunk → ClassResult (cnn_attn_full ONNX 추론, 5초 롤링 버퍼)."""
    return _CLF.infer(chunk)


def analyze(chunk: AudioChunk):
    """하이브리드: 마진(5s 확정+2s 예비) dict. 워밍업이면 None."""
    return _CLF.analyze(chunk)


def subtype_probs():
    """하이브리드: 마지막 창의 차종 확률[3] | None."""
    return _CLF.subtype_probs()


def speed_evidence():
    """완전한 5초 창의 속도 모델 원시 증거 | None."""
    return _CLF.speed_evidence()


def reset() -> None:
    """이벤트 경계에서 버퍼 초기화."""
    _CLF.reset()
