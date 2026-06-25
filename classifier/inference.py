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
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from core.types import AudioChunk, ClassResult, SoundClass

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
        return ClassResult.from_label(_TO_SOUNDCLASS[CLASSES[i]], float(prob[i]))


_CLF = _OnnxClassifier()


def infer(chunk: AudioChunk) -> ClassResult:
    """AudioChunk → ClassResult (cnn_attn_full ONNX 추론, 5초 롤링 버퍼)."""
    return _CLF.infer(chunk)


def reset() -> None:
    """이벤트 경계에서 버퍼 초기화."""
    _CLF.reset()
