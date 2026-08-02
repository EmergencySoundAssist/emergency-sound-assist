"""
교체 가능한 VAD(음성 활동 감지).

energy(RMS) 게이트는 '소리 큼'만 봐서 노이즈를 음성으로 오인 → Whisper 환각을 유발했다.
기본은 faster-whisper에 포함된 Silero ONNX VAD다. 공개 도로소음 검증에서 WebRTC보다
발화 경계를 잘 구분했고, torch 없이 동작한다. Silero를 불러올 수 없으면 WebRTC, 그마저
없으면 energy 게이트로 폴백한다.

인터페이스: vad.is_speech(samples_float32_mono, sample_rate) -> bool
교체점: 실차 데이터에서 부족하면 같은 인터페이스로 다른 VAD를 추가한다.
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


class SileroVad:
    """faster-whisper 내장 Silero ONNX VAD를 1초 청크 게이트로 사용한다."""

    def __init__(self, threshold: float = 0.5, voiced_ratio: float = 0.2,
                 _timestamps=None):
        if _timestamps is None:
            from faster_whisper.vad import VadOptions, get_speech_timestamps, get_vad_model

            get_vad_model()  # onnxruntime까지 지금 확인해 auto 폴백이 실제로 동작하게 한다.
            self._timestamps = get_speech_timestamps
            self._options = VadOptions(
                threshold=threshold,
                min_speech_duration_ms=64,
                min_silence_duration_ms=100,
                speech_pad_ms=0,
            )
        else:
            self._timestamps = _timestamps
            self._options = None
        self._voiced_ratio = voiced_ratio

    def is_speech(self, x: np.ndarray, sample_rate: int) -> bool:
        if sample_rate != 16000 or x.size == 0:
            return _rms(x) >= 0.02
        segments = self._timestamps(
            np.asarray(x, dtype=np.float32), self._options, sampling_rate=sample_rate
        )
        voiced = sum(max(0, int(item["end"]) - int(item["start"])) for item in segments)
        return (voiced / len(x)) >= self._voiced_ratio


def make_vad(cfg):
    """설정에 따라 VAD 선택. auto = Silero → WebRTC → energy."""
    backend = getattr(cfg, "vad_backend", "auto")
    if backend not in ("auto", "silero", "webrtc", "energy"):
        raise ValueError(f"알 수 없는 VAD backend: {backend}")
    if backend in ("auto", "silero"):
        try:
            vad = SileroVad(
                cfg.silero_threshold, voiced_ratio=cfg.silero_voiced_ratio
            )
            print("[stt] VAD: silero-onnx", file=sys.stderr)
            return vad
        except Exception as e:
            if backend == "silero":
                raise
            print(f"[stt] Silero VAD 없음 → WebRTC VAD 폴백 ({e})", file=sys.stderr)
    if backend in ("auto", "webrtc"):
        try:
            vad = WebRtcVad(cfg.webrtc_aggressiveness, voiced_ratio=cfg.webrtc_voiced_ratio)
            print("[stt] VAD: webrtcvad", file=sys.stderr)
            return vad
        except Exception as e:                # webrtcvad 미설치 등
            if backend == "webrtc":
                raise
            print(f"[stt] WebRTC VAD 없음 → energy VAD 폴백 ({e})", file=sys.stderr)
    return EnergyVad(cfg.vad_rms_threshold)
