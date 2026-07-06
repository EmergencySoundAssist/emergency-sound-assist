"""오디오 파일 I/O 공용 유틸 — 항상 16kHz mono float32 로 통일.

soundfile(libsndfile≥1.1)은 wav/flac 은 물론 mp3 도 읽는다(edge-tts 출력).
시스템에 ffmpeg 이 없어도 동작해야 하므로 여기서만 soundfile+soxr 을 쓴다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

SR = 16000


def load_mono_16k(path) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SR:
        data = soxr.resample(data, sr, SR)
    return np.asarray(data, dtype=np.float32)


def save_wav_16k(path, samples: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(samples, dtype=np.float32), SR, subtype="PCM_16")


def crop_or_tile(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """길이 n 짜리 조각을 만든다. 길면 랜덤 크롭, 짧으면 반복해서 채운 뒤 크롭."""
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0:
        raise ValueError("빈 오디오")
    if len(x) < n:
        x = np.tile(x, int(np.ceil(n / len(x))))
    start = int(rng.integers(0, len(x) - n + 1))
    return x[start: start + n].copy()
