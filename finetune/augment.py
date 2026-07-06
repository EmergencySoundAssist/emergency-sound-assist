"""평가셋 증강 — SNR 노이즈 믹싱·확성기 시뮬. 전부 순수 함수(파일 I/O 없음).

확성기 근사 근거: 전화/PA 대역(300–3400Hz) 밴드패스 + 소프트클립(과구동 왜곡)
+ 30ms 단일 반사(야외 잔향 최소 근사). 실측 확성기 IR 이 아니라 근사임 —
평가 결과 해석 시 유의 (docs/stt/finetune.md 한계 참고).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt


def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """speech 에 noise 를 목표 SNR(dB)로 섞는다. noise 는 speech 이상 길이여야 한다.

    믹스 결과 피크가 1.0 을 넘으면 전체를 스케일다운한다(SNR 은 유지, 클리핑 방지).
    """
    speech = np.asarray(speech, dtype=np.float32)
    noise = np.asarray(noise, dtype=np.float32)
    if len(noise) < len(speech):
        raise ValueError(f"noise({len(noise)}) 가 speech({len(speech)}) 보다 짧음")
    noise = noise[: len(speech)]

    s_rms, n_rms = _rms(speech), _rms(noise)
    if s_rms < 1e-8 or n_rms < 1e-8:      # 무음이면 섞을 의미 없음
        return speech.copy()

    gain = s_rms / (n_rms * 10.0 ** (snr_db / 20.0))
    mixed = speech + noise * gain
    peak = float(np.max(np.abs(mixed)))
    if peak > 1.0:
        mixed = mixed / peak
    return mixed.astype(np.float32)


def simulate_loudspeaker(x: np.ndarray, sr: int = 16000) -> np.ndarray:
    """확성기/PA 근사: 300–3400Hz 밴드패스 → 소프트클립 → 30ms 반사 1개."""
    x = np.asarray(x, dtype=np.float32)
    sos = butter(4, [300.0, 3400.0], btype="bandpass", fs=sr, output="sos")
    y = sosfilt(sos, x).astype(np.float32)

    y = (np.tanh(3.0 * y) / np.tanh(3.0)).astype(np.float32)   # drive=3 과구동

    d = int(0.030 * sr)
    if 0 < d < len(y):
        echo = np.zeros_like(y)
        echo[d:] = y[:-d] * 0.3
        y = y + echo

    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1.0:
        y = y / peak
    return y.astype(np.float32)
