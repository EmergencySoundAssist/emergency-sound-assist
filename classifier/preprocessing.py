"""
전처리: 오디오 파형 → 모델 입력.

두 종류의 입력을 만든다.
- torch 모델(scratch CNN / MobileNet): **log-Mel spectrogram** (1, n_mels, time)
- YAMNet: **16kHz 파형 그대로** (YAMNet이 내부에서 mel/임베딩 생성)

학습(dataset.py)과 추론(infer.py)이 동일하게 이 모듈을 써서
"학습 때와 추론 때 전처리가 달라지는" 버그를 방지한다.
"""

from __future__ import annotations

import numpy as np
import torch
import torchaudio

from core.types import AudioChunk
from . import config

# ---------------------------------------------------------------------------
# Mel-spectrogram 변환기 (config 값으로 1번만 생성해 재사용)
# ---------------------------------------------------------------------------
_MEL = torchaudio.transforms.MelSpectrogram(
    sample_rate=config.SAMPLE_RATE,
    n_fft=config.N_FFT,
    hop_length=config.HOP_LENGTH,
    n_mels=config.N_MELS,
)
_TO_DB = torchaudio.transforms.AmplitudeToDB(stype="power")


# ---------------------------------------------------------------------------
# 공통 유틸
# ---------------------------------------------------------------------------
def _as_mono_tensor(samples: np.ndarray | torch.Tensor) -> torch.Tensor:
    """입력을 float32 모노 1D 텐서로 변환."""
    if isinstance(samples, np.ndarray):
        wav = torch.from_numpy(np.ascontiguousarray(samples)).float()
    else:
        wav = samples.float()
    if wav.ndim == 2:                      # (n, channels) → 모노 다운믹스
        wav = wav.mean(dim=1)
    return wav.reshape(-1)


def _fix_length(wav: torch.Tensor, n: int = config.CHUNK_SAMPLES) -> torch.Tensor:
    """길이를 정확히 n 샘플로 맞춤 (길면 자르고, 짧으면 0으로 패딩)."""
    if wav.numel() >= n:
        return wav[:n]
    pad = n - wav.numel()
    return torch.nn.functional.pad(wav, (0, pad))


def _resample(wav: torch.Tensor, src_sr: int) -> torch.Tensor:
    """필요 시 config.SAMPLE_RATE 로 리샘플."""
    if src_sr == config.SAMPLE_RATE:
        return wav
    return torchaudio.functional.resample(wav, src_sr, config.SAMPLE_RATE)


def prepare_waveform(
    samples: np.ndarray | torch.Tensor,
    sample_rate: int = config.SAMPLE_RATE,
) -> torch.Tensor:
    """오디오 → 16kHz 모노, 1초 고정 길이 파형 (n,). YAMNet 입력용."""
    wav = _as_mono_tensor(samples)
    wav = _resample(wav, sample_rate)
    wav = _fix_length(wav)
    return wav


# ---------------------------------------------------------------------------
# torch 모델용: log-Mel spectrogram
# ---------------------------------------------------------------------------
def waveform_to_logmel(
    samples: np.ndarray | torch.Tensor,
    sample_rate: int = config.SAMPLE_RATE,
) -> torch.Tensor:
    """오디오 → log-Mel spectrogram 텐서 (1, n_mels, time).

    scratch CNN / MobileNet 입력. (MobileNet의 3채널 변환은 모델 쪽에서 처리)
    """
    wav = prepare_waveform(samples, sample_rate)     # (n,)
    mel = _MEL(wav)                                  # (n_mels, time)
    logmel = _TO_DB(mel)                             # dB 스케일
    logmel = _normalize(logmel)                      # 정규화
    return logmel.unsqueeze(0)                       # (1, n_mels, time)


def _normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """샘플 단위 표준화 (평균0/분산1). 녹음 음량 차이에 강건해짐."""
    return (x - x.mean()) / (x.std() + eps)


# ---------------------------------------------------------------------------
# AudioChunk 래퍼 (추론 파이프라인에서 편하게)
# ---------------------------------------------------------------------------
def chunk_to_logmel(chunk: AudioChunk) -> torch.Tensor:
    return waveform_to_logmel(chunk.samples, chunk.sample_rate)


def chunk_to_waveform(chunk: AudioChunk) -> torch.Tensor:
    return prepare_waveform(chunk.samples, chunk.sample_rate)
