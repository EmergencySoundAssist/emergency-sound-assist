"""
공통 오디오 입력 (모든 모듈이 공유).

노트북 개발 단계에서는 노트북 내장 마이크나 WAV 파일로 테스트하고,
나중에 Jetson + ReSpeaker 로 옮길 때 이 파일의 입력 소스만 바꾸면 된다.

의존성: sounddevice(실시간 마이크), soundfile(파일). 둘 다 선택 설치.
파일 재생/테스트만 할 거면 sounddevice 없이도 load_wav() 사용 가능.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from core.types import AudioChunk, SAMPLE_RATE, CHUNK_SECONDS


def load_wav(path: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """WAV 파일을 (n_samples,) float32 모노로 로드. 분류 테스트용."""
    import soundfile as sf  # 지연 import: 파일 안 쓰면 설치 불필요

    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:                    # 다채널이면 평균내서 모노로
        data = data.mean(axis=1)
    if sr != sample_rate:
        data = _resample(data, sr, sample_rate)
    return data


def iter_chunks_from_array(
    samples: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    chunk_seconds: float = CHUNK_SECONDS,
) -> Iterator[AudioChunk]:
    """긴 오디오 배열을 1초짜리 청크들로 잘라서 내보낸다(파일 테스트용)."""
    n = int(sample_rate * chunk_seconds)
    for start in range(0, len(samples) - n + 1, n):
        yield AudioChunk(samples=samples[start:start + n], sample_rate=sample_rate)


def iter_chunks_from_mic(
    sample_rate: int = SAMPLE_RATE,
    chunk_seconds: float = CHUNK_SECONDS,
    device: int | None = None,
) -> Iterator[AudioChunk]:
    """실시간 마이크에서 1초씩 읽어 청크로 내보낸다.

    노트북: 내장 마이크 (device=None).
    Jetson: ReSpeaker 장치 인덱스를 device 로 지정.
    """
    import sounddevice as sd  # 지연 import

    n = int(sample_rate * chunk_seconds)
    with sd.InputStream(samplerate=sample_rate, channels=1,
                        dtype="float32", device=device) as stream:
        while True:
            data, _ = stream.read(n)
            yield AudioChunk(samples=data[:, 0].copy(), sample_rate=sample_rate)


def _resample(data: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """간단 선형 리샘플링(테스트용). 정밀 작업은 librosa.resample 권장."""
    duration = len(data) / src_sr
    dst_len = int(duration * dst_sr)
    x_old = np.linspace(0.0, duration, num=len(data), endpoint=False)
    x_new = np.linspace(0.0, duration, num=dst_len, endpoint=False)
    return np.interp(x_new, x_old, data).astype(np.float32)
