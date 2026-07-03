"""
교체 가능한 VAD(음성 활동 감지).

energy(RMS) 게이트는 '소리 큼'만 봐서 노이즈를 음성으로 오인 → Whisper 환각을 유발했다.
webrtcvad(Google WebRTC VAD)는 진짜 '말소리/비음성'을 구분해 노이즈를 잘 거른다.
torch 의존성이 없어 Jetson(ONNX-only 런타임)에도 부담이 없다(RealtimeSTT 의 1차 VAD 와 동일).

인터페이스: vad.is_speech(samples_float32_mono, sample_rate) -> bool
교체점: 더 강한 노이즈 robust 가 필요하면 여기에 SileroOnnxVad(onnxruntime 직접) 추가.
"""

from __future__ import annotations

import sys

import numpy as np


def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def _to_pcm16(x: np.ndarray) -> bytes:
    """float32[-1,1] 모노 → 16-bit PCM 바이트(webrtcvad 입력 형식)."""
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


class EnergyVad:
    """RMS 에너지 게이트(의존성 0). webrtcvad 없을 때 폴백."""

    def __init__(self, threshold: float):
        self._threshold = threshold

    def is_speech(self, x: np.ndarray, sample_rate: int) -> bool:
        return _rms(x) >= self._threshold


class WebRtcVad:
    """WebRTC VAD. 30ms 프레임 단위로 음성 판정 후, 청크 내 음성 프레임 비율로 결정."""

    def __init__(self, aggressiveness: int = 2, voiced_ratio: float = 0.2,
                 frame_ms: int = 30, _impl=None):
        if _impl is None:
            import webrtcvad  # 지연 import: 없으면 make_vad 가 energy 로 폴백
            _impl = webrtcvad.Vad(aggressiveness)
        self._vad = _impl
        self._voiced_ratio = voiced_ratio
        self._frame_ms = frame_ms

    def is_speech(self, x: np.ndarray, sample_rate: int) -> bool:
        # webrtcvad 는 8/16/32/48kHz 만 지원. 그 외면 에너지로 대충.
        if sample_rate not in (8000, 16000, 32000, 48000):
            return _rms(x) >= 0.02
        pcm = _to_pcm16(x)
        frame_len = int(sample_rate * self._frame_ms / 1000)   # 샘플 수
        fb = frame_len * 2                                     # int16 바이트
        n = len(pcm) // fb
        if n == 0:
            return False
        voiced = 0
        for i in range(n):
            if self._vad.is_speech(pcm[i * fb:(i + 1) * fb], sample_rate):
                voiced += 1
        return (voiced / n) >= self._voiced_ratio


def make_vad(cfg):
    """설정에 따라 VAD 선택. auto = webrtcvad 있으면 사용, 없으면 energy."""
    backend = getattr(cfg, "vad_backend", "auto")
    if backend in ("auto", "webrtc"):
        try:
            vad = WebRtcVad(cfg.webrtc_aggressiveness, voiced_ratio=cfg.webrtc_voiced_ratio)
            print("[stt] VAD: webrtcvad", file=sys.stderr)
            return vad
        except Exception as e:                # webrtcvad 미설치 등
            if backend == "webrtc":
                raise
            print(f"[stt] webrtcvad 없음 → energy VAD 폴백 ({e})", file=sys.stderr)
    return EnergyVad(cfg.vad_rms_threshold)
