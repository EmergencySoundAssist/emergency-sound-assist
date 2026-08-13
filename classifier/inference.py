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
# 실차 파인튜닝판(_early4)이 기본이다. 런타임 경로 실측(워밍업 제외, 실차 사이렌 17클립):
#            실차지연  ★조기검출  어제오탐   8/12밤오탐  놓침
#   구모델      5.0초     6.8%    44/49      5/59     0/17
#   early3      2.0초    86.4%    11/49     46/59     0/17
#   early4      2.0초    81.4%    13/49      9/59     0/17   ← 기본
# early3 이 8/12 밤 현장에서 46/59 오탐을 냈고(같은 장소·같은 조건), 그 클립들을
# 네거티브로 학습해 9/59 로 줄인 것이 early4 다. 지연·놓침은 그대로.
# ⚠ '8/12밤 오탐'은 early3 가 울려 수집된 클립이라 early3 에겐 순환(불리), early4 엔
#   학습셋(유리)이다. 공평한 잣대는 '어제 오탐'이고 거기선 11→13 으로 소폭 나빠졌다.
# rc5 는 여기서 한 걸음 더 간다 — 실차 사이렌이 19개뿐인 게 병목이라, 그 19개로 **채널만
# 실측해**(realchannel.npz: 사이렌 대역 EQ -2.63, 실차 소음 32분) AI-Hub 사이렌 2,239개에
# 입혔다. 라벨은 AI-Hub 에서, 채널은 실차에서 가져오는 셈이다.
# 대조군(ctl5 = 같은 레시피, 채널만 뺌)과 비교하면 게이트 지점마다 지연·오탐 양쪽을 이긴다:
#   0.25초 격자 τ=+0.4 →  ctl5 3.50초 / 오탐 13클립   vs   rc5 2.88초 / 오탐 8클립
# ⚠ 근거의 한계: 벤치 양성은 AI-Hub(실차 아님)이고, 네거티브 142클립은 두 모델 다
#   학습에 본 것이라 절대값은 낙관적이다. 실차 사이렌 19개 k-fold(제대로 홀드아웃)에서는
#   조기검출 틱이 오히려 72.6%→70.1% 로 조금 내려갔다(오발화는 3.7%→3.2%). 부호가 갈린다.
# rc5 는 **아직 기본이 아니다** — 현장 재검증 전이다. 켜려면 게이트·격자와 셋을 같이:
#   ESA_DETECT_MODEL=cnn_attn_full_s42_rc5.onnx ESA_GATE=1  (+ CHUNK_SECONDS=0.25)
# 구모델로 되돌리려면 ESA_DETECT_MODEL=cnn_attn_full_s42.onnx 로 실행한다.
_MODEL_PATH = _MODELS / os.environ.get("ESA_DETECT_MODEL", "cnn_attn_full_s42_early4.onnx")

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
# 예비 창은 확정 창과 **같은 가중치**여야 한다. 검출 모델 이름에서 87프레임 판을
# 유도하고, 없으면 기본판으로 떨어진다. (하드코딩하면 새 모델을 ESA_DETECT_MODEL 로
# 물렸을 때 5초 창만 새것 · 2초 창은 옛것이 되어 두 창이 서로 다른 모델이 된다.)
def _fast_path_for(detect: Path) -> Path:
    sibling = detect.with_name(detect.stem + "_87f.onnx")
    return sibling if sibling.exists() else _MODELS / "cnn_attn_full_s42_87f.onnx"


_FAST_PATH = _fast_path_for(_MODEL_PATH)
FAST_FRAMES = 87

# ── 마진 게이트 (선택) ────────────────────────────────────────────────
# argmax(마진>0) 대신 pipeline.gate 의 히스테리시스+투표+hangover 로 판정한다.
# CHUNK_SECONDS=0.25 와 **짝**이다 — 굵은 격자에서는 투표창이 1틱으로 붕괴해 게이트가
# 단순 임계로 전락하고, 게이트 없는 촘촘한 격자는 오탐이 폭증한다(현장에서 겪었다).
# τ=+0.4 를 고른 이유: 현재 배포 지점(1.0초 격자·게이트 없음 = 지연 3.00초/오탐 22클립)을
# **양쪽 축에서** 이기는 구간(τ=0.0~0.4) 중 오탐이 가장 낮은 끝이다 → 2.75초 / 12클립.
# 기본은 꺼짐 — 현장 재검증 전이다. ESA_GATE=1 로 켜고, 켤 때는 CHUNK_SECONDS=0.25 와 함께.
GATE_ON = os.environ.get("ESA_GATE", "0") not in ("0", "false", "False")
GATE_TAU_ON = float(os.environ.get("ESA_GATE_TAU_ON", "0.4"))


def emergency_from(res) -> bool:
    """analyze() 결과 → 긴급인가 (게이트 **없는** 기본 경로의 판정 규칙).

    5초 창 argmax 가 siren/horn 이거나, 5초 창이 noise 인데 2초 창이 horn 이면 긴급이다.
    두 번째 항이 핵심인데, 벤치가 이걸 안 세고 있었다 — 같은 early4·같은 142 네거티브·
    1.0초 격자에서 벤치 규칙 22클립 vs 런타임 규칙 25클립. 배포 판단의 기준선이었던
    그 22 가 실제 동작보다 3클립 낮았다. 그래서 규칙을 여기 한 곳에 둔다.
    """
    if res is None:
        return False
    if res["label"] in (SoundClass.SIREN, SoundClass.HORN):
        return True
    return (res["label"] is SoundClass.NORMAL_TRAFFIC
            and res.get("fast_label") is SoundClass.HORN)


def gate_margins(res) -> tuple[float, float]:
    """analyze() 결과 → 게이트에 넣을 (사이렌, 경적) 마진.

    경적은 5초 창과 2초 창 중 **큰 쪽**을 쓴다. 경적은 AI-Hub 기준 중앙 2.8초라
    5초 창에서는 창을 절반도 못 채우고 창 단위 정규화가 그마저 희석한다 —
    5초 마진만 보면 늦게 뜨는 게 아니라 아예 놓친다(실측 43/57 → 52/57).
    사이렌은 2초 창을 열면 오탐만 늘고 지연은 그대로라(11→23/49) 열지 않는다.

    벤치(tools/bench_runtime.py)와 런타임이 **같은 함수**를 부른다 — 따로 두면
    재본 적 없는 규칙을 배포하게 된다.
    """
    fh = res.get("m_fast_horn")
    m_horn = res["m_horn"] if fh is None else max(res["m_horn"], fh)
    return res["m_siren"], m_horn


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
        self._buf_in = np.zeros(0, dtype=np.float32)   # 입력 레이트 원본 버퍼
        self._buf_sr = 0
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
        self._gate = None                    # 마진 게이트 (ESA_GATE=1 일 때만)
        self._gate_dt = 0.0                  # 게이트를 만든 청크 간격(초)

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
        self._buf_in = np.zeros(0, dtype=np.float32)
        self._last_x = None
        if self._gate is not None:
            self._gate.reset()      # 클립 경계에서 hangover 가 넘어가면 오프라인 재생이 오염된다

    # ── 하이브리드 API — 마진 판정 + 창 재사용 (석우 infer_trt.step 방식) ──
    def analyze(self, chunk: AudioChunk):
        """청크 누적 → 멜 1회 → 검출 로짓 **마진**(5초 확정 + 2초 예비).

        softmax(1.000 포화) 대신 마진 z[cls]-max(나머지)로 게이트 판정(alert.Gate 입력).
        정규화된 5초 창을 보관(self._last_x) → subtype_probs()/speed_evidence()가 재사용(멜 1회).
        반환: dict(label, conf, m_siren, m_horn, m_fast|None) — 버퍼 부족(워밍업)이면 None.
        """
        # 버퍼는 **입력 레이트 그대로** 들고, 평가할 때 창 전체를 한 번에 리샘플한다.
        # 청크마다 따로 리샘플해 이어 붙이면 조각 경계마다 FIR 과도응답이 남아,
        # 청크가 작을수록(0.25초) 이음매가 잦아져 판정이 흔들린다(실측: 같은 오디오인데
        # 1.0초 격자 최대 지연 4.0초 → 0.25초 격자 7.0초로 악화). 이렇게 두면 결과가
        # **청크 크기와 무관**해진다 — 격자를 자유롭게 촘촘하게 할 수 있다.
        sr_in = int(chunk.sample_rate)
        if sr_in != self._buf_sr:                # 입력 레이트가 바뀌면 버퍼를 새로 시작
            self._buf_in = np.zeros(0, dtype=np.float32)
            self._buf_sr = sr_in
        keep = int(WIN_S * sr_in)
        self._buf_in = np.concatenate([self._buf_in, _mono(chunk.samples)])[-keep:]
        self._buf = _resample(self._buf_in, sr_in, SR_MODEL)
        # ★ 창이 실제 오디오로 다 차기 전에는 판정하지 않는다.
        #
        # 버퍼가 짧으면 아래에서 무음(PAD_VAL=log 1e-6)으로 216프레임을 채우는데,
        # _norm_win 이 그 계단까지 포함해 정규화하므로 실제 오디오 쪽이 통째로 과장된다.
        # 실측(early4·게이트 off·실차 네거티브 127클립, 런타임 규칙 infer 로 셈):
        #     t<5s  : 틱 331/508 (65.2%) 발화 · **클립 127/127**
        #     t>=5s : 틱  65/1340 ( 4.9%) 발화 · 클립  25/127
        # 즉 켤 때마다 반드시 한 번은 사이렌 경보와 BLE 진동이 나갔다.
        # Airacle deploy 도 같은 이유로 막는다 — infer_trt.py `if not ring.full(): continue`.
        #
        # 대가: 기동 후 5초 안에 시작한 사이렌은 놓친다. 매 기동 오경보를 확실히 내는
        # 것보다 낫다고 본다(deploy 도 같은 선택). tools/bench_runtime.py 가 t<5.0 틱을
        # 버리는 것도 이제서야 사실과 맞는다.
        if self._buf.size + HOP < BUF_SAMPLES:
            return None
        if self._buf.size < N_FFT:               # 방어: 창 하나도 못 만드는 길이
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
        m_fast_horn = None
        fast_label = None
        fs = self._fast_session()
        if fs is not None:                        # 예비: 같은 멜의 끝 2초만, 별도 정규화
            xf = np.ascontiguousarray(_norm_win(m[:, -FAST_FRAMES:])[None, None], dtype=np.float32)
            zf = fs.run(None, {self._fast_in: xf})[0][0]
            m_fast = float(zf[0] - max(zf[1], zf[2]))
            # 경적 마진도 같이 낸다 — 게이트는 마진으로만 판정하는데, 경적은 5초 창에서
            # 창의 절반도 못 채워(중앙 2.8초) 5초 마진만 보면 **아예 놓친다**.
            m_fast_horn = float(zf[1] - max(zf[0], zf[2]))
            fast_label = _TO_SOUNDCLASS[CLASSES[int(np.argmax(zf))]]

        self._last_x = x
        return {"label": _TO_SOUNDCLASS[CLASSES[i]], "conf": float(prob[i]),
                "m_siren": m_siren, "m_horn": m_horn, "m_fast": m_fast,
                "m_fast_horn": m_fast_horn, "fast_label": fast_label}

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
        """분류 단독 API. 사이렌이면 동일 5초 창에서 차종도 함께 반환한다.

        **경적만 2초 창을 함께 본다.** 경적은 짧다 — AI-Hub 기준 중앙 2.8초이고
        65%가 5초 미만이라, 5초 창에서는 경적이 창의 절반도 못 채우고 창 단위
        정규화가 그마저 희석한다. 그래서 경적은 '늦게 뜨는' 게 아니라 **아예
        놓친다**(실측: 5초 창 43/57 → 2초 창을 더하면 52/57, 지연은 둘 다 1.0초).
        대가는 실차 네거티브 11/49 → 13/49, 음악 3/12 → 3/12 로 작다.

        사이렌까지 2초 창을 열면(전체 OR) 네거티브가 11 → 23/49 로 뛰는데 사이렌
        지연은 2.0초 그대로라 이득이 없다. 그래서 **경적에만** 연다.
        """
        res = self.analyze(chunk)
        if res is None:
            return ClassResult.from_label(SoundClass.NORMAL_TRAFFIC, 0.0)
        if GATE_ON:
            return self._gated(res, chunk)
        if res["label"] is SoundClass.NORMAL_TRAFFIC and emergency_from(res):
            return ClassResult.from_label(SoundClass.HORN, res["conf"])   # 2초 창 경적 승격
        subtype = subtype_confidence = None
        if res["label"] is SoundClass.SIREN and self._last_x is not None:
            subtype, subtype_confidence = self._infer_subtype(self._last_x)
        return ClassResult.from_label(
            res["label"],
            res["conf"],
            subtype=subtype,
            subtype_confidence=subtype_confidence,
        )

    def _gated(self, res, chunk: AudioChunk) -> ClassResult:
        """마진 게이트로 판정한다 — argmax(마진>0) 대신 켜기 어렵게·끄기 느리게.

        게이트의 시간상수는 **초**라, 청크 간격이 바뀌면 틱 수로 다시 환산해야 한다.
        간격이 달라지면 게이트를 새로 만든다(중간에 격자를 바꿔도 실효 동작 유지).

        사이렌/경적 중 어느 쪽이 켰는지로 라벨을 정한다. 게이트가 꺼져 있으면 마진과
        무관하게 '일반 도로 소음' — 고립된 튐이 경보로 새지 않는다.
        """
        dt = len(chunk.samples) / float(chunk.sample_rate or 1)
        if self._gate is None or abs(dt - self._gate_dt) > 1e-6:
            from pipeline.gate import EmergencyGate, configs_for
            sc, hc = configs_for(GATE_TAU_ON)    # 벤치(tools/bench_runtime.py)와 같은 코드
            self._gate = EmergencyGate(dt, siren=sc, horn=hc)
            self._gate_dt = dt
        on = self._gate.step(*gate_margins(res))
        if not on:
            return ClassResult.from_label(SoundClass.NORMAL_TRAFFIC, res["conf"])
        siren = self._gate.siren.on
        if not siren:
            return ClassResult.from_label(SoundClass.HORN, res["conf"])
        subtype = subtype_confidence = None
        if self._last_x is not None:
            subtype, subtype_confidence = self._infer_subtype(self._last_x)
        return ClassResult.from_label(SoundClass.SIREN, res["conf"],
                                      subtype=subtype,
                                      subtype_confidence=subtype_confidence)


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
