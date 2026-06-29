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
_MODEL_PATH = Path(__file__).resolve().parent / "models" / "cnn_attn_full_s42.onnx"

# ── 차종(사이렌 세분화) — ViT subtype_clf 모델 (선택) ────────────────
# 검출과 입력(64×216 멜)·정규화가 동일 → 같은 윈도우를 그대로 재사용한다.
# 모델 파일이 없으면 차종은 생략하고 기존처럼 'siren'으로만 출력(graceful).
SUBS = ("ambulance", "police", "fire")        # subtype_clf.SUBS=[구급차,경찰차,소방차] 순서
_TO_SUBTYPE = {
    "ambulance": SirenSubtype.AMBULANCE,
    "police": SirenSubtype.POLICE,
    "fire": SirenSubtype.FIRE,
}
_SUBTYPE_PATH = Path(__file__).resolve().parent / "models" / "subtype_cnn_attn_s42.onnx"
SUBTYPE_CONF = 0.6        # 미만이면 '긴급차량'(UNKNOWN)으로 일반화 (경찰↔구급 혼동 회피)

# ── 접근 속도(사이렌 빠르기) — ViT speed_neural 모델 (선택) ──────────
# 검출과 입력(64×216 멜, hop 512)·정규화가 동일 → 같은 윈도우를 그대로 재사용한다.
# 출력은 km/h(0~80 학습) → 1~5단계로 변환. ⚠ 실주행 미검증(정지 환각) — 데모용.
_SPEED_PATH = Path(__file__).resolve().parent / "models" / "speed_neural.onnx"


def _kmh_to_level(v_kmh: float) -> int:
    """추정 km/h → 접근 속도 1~5단계 (16km/h 폭, 0~80 학습 범위)."""
    return int(min(5, max(1, int(v_kmh // 16) + 1)))


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

    def _infer_speed(self, x: np.ndarray):
        """정규화된 멜 (1,1,64,216) → (속도 1~5단계, km/h) | (None, None).
        검출과 동일 입력이라 같은 윈도우 x 를 그대로 넣는다. 사이렌일 때만 호출."""
        sess = self._speed_session()
        if sess is None:
            return None, None
        out = sess.run(None, {self._spd_in: x})       # [speed(km/h), f0]
        v = max(0.0, float(np.asarray(out[0]).ravel()[0]))
        return _kmh_to_level(v), v

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)

    def infer(self, chunk: AudioChunk) -> ClassResult:
        y = _resample(_mono(chunk.samples), chunk.sample_rate, SR_MODEL)
        self._buf = np.concatenate([self._buf, y])[-BUF_SAMPLES:]
        if self._buf.size < N_FFT:               # 아직 너무 짧음
            return ClassResult.from_label(SoundClass.NORMAL_TRAFFIC, 0.0)

        m = _logmel(self._buf)[:, -N_FRAMES:]    # 최근 5초 윈도우
        if m.shape[1] < N_FRAMES:                # 버퍼가 5초 미만이면 끝을 무음 패딩
            m = np.pad(m, ((0, 0), (0, N_FRAMES - m.shape[1])), constant_values=PAD_VAL)
        x = (m - m.mean()) / (m.std() + 1e-5)    # 윈도우별 정규화
        x = np.ascontiguousarray(x[None, None], dtype=np.float32)   # (1,1,64,216)

        logits = self._session().run(None, {self._in: x})[0][0]
        prob = _softmax(logits)
        i = int(prob.argmax())
        label = _TO_SOUNDCLASS[CLASSES[i]]

        subtype, sub_conf = None, None
        speed_level, speed_kmh = None, None
        if label is SoundClass.SIREN:            # 차종·속도는 사이렌일 때만 (멜 그대로 재사용)
            subtype, sub_conf = self._infer_subtype(x)
            speed_level, speed_kmh = self._infer_speed(x)
        return ClassResult.from_label(label, float(prob[i]), subtype, sub_conf,
                                      speed_level, speed_kmh)


_CLF = _OnnxClassifier()


def infer(chunk: AudioChunk) -> ClassResult:
    """AudioChunk → ClassResult (cnn_attn_full ONNX 추론, 5초 롤링 버퍼)."""
    return _CLF.infer(chunk)


def reset() -> None:
    """이벤트 경계에서 버퍼 초기화."""
    _CLF.reset()
