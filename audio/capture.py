"""
공통 오디오 입력 (모든 모듈이 공유).

노트북 개발 단계에서는 노트북 내장 마이크나 WAV 파일로 테스트하고,
나중에 Jetson + ReSpeaker 로 옮길 때 이 파일의 입력 소스만 바꾸면 된다.

의존성: sounddevice(실시간 마이크), soundfile(파일). 둘 다 선택 설치.
파일 재생/테스트만 할 거면 sounddevice 없이도 load_wav() 사용 가능.
"""

from __future__ import annotations

import sys
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
    channels: int = 1,
) -> Iterator[AudioChunk]:
    """실시간 마이크에서 1초씩 읽어 청크로 내보낸다.

    노트북: 내장 마이크 1채널 (channels=1, device=None).
    Jetson: ReSpeaker 6채널 (channels=6, device=인덱스) → ch0=분류·접근, ch1~4=방향.

    channels==1 이면 (n,) 모노, 그 이상이면 (n, channels) 다채널을 그대로 담는다.
    (분류·접근은 pipeline 이 ch0 만, 방향은 ch1~4 만 골라 쓴다.)
    """
    import sounddevice as sd  # 지연 import

    n = int(sample_rate * chunk_seconds)
    with sd.InputStream(samplerate=sample_rate, channels=channels,
                        dtype="float32", device=device) as stream:
        while True:
            data, _ = stream.read(n)
            samples = data[:, 0].copy() if channels == 1 else data.copy()
            yield AudioChunk(samples=samples, sample_rate=sample_rate)


def iter_chunks_from_respeaker(
    channel: int = 0,
    num_channels: int = 6,
    sample_rate: int = SAMPLE_RATE,
    chunk_seconds: float = CHUNK_SECONDS,
    device: int | None = None,
) -> Iterator[AudioChunk]:
    """Jetson + ReSpeaker 용 캡처. 6채널로 열어 한 채널만 모노로 내보낸다.

    ReSpeaker XVF-3000 은 USB 로 6채널을 준다(→ docs/hardware.md):
      ch0 = 빔포밍/AEC 처리된 깨끗한 1채널  ← STT 에 가장 적합(기본값)
      ch1~4 = 원본 마이크 / ch5 = 재생 참조
    device=None 이면 'ReSpeaker' 가 이름에 들어간 장치를 자동 탐지한다.
    """
    import sounddevice as sd  # 지연 import

    if device is None:
        device = _find_respeaker_index()

    n = int(sample_rate * chunk_seconds)
    with sd.InputStream(samplerate=sample_rate, channels=num_channels,
                        dtype="float32", device=device) as stream:
        while True:
            data, _ = stream.read(n)          # (n, num_channels)
            yield AudioChunk(samples=data[:, channel].copy(), sample_rate=sample_rate)


def iter_chunks_threaded(source: Iterator, maxsize: int = 120) -> Iterator:
    """source(청크 제너레이터)를 백그라운드 스레드에서 읽어 큐로 흘려보낸다.

    소비자가 변환(블로킹)하는 동안에도 캡처 스레드는 계속 마이크를 읽으므로
    '변환 중 입력 못 받음'을 막는다. 큐가 maxsize(기본 ~2분) 차면 producer 가 대기.
    """
    import threading
    import queue

    q: "queue.Queue" = queue.Queue(maxsize=maxsize)
    sentinel = object()

    def _producer():
        try:
            for item in source:
                q.put(item)
        except Exception as e:          # 캡처 에러를 소비자 쪽으로 전달
            q.put(e)
        finally:
            q.put(sentinel)

    threading.Thread(target=_producer, daemon=True).start()
    while True:
        item = q.get()
        if item is sentinel:
            break
        if isinstance(item, Exception):
            raise item
        yield item


def _find_respeaker_index() -> int | None:
    """입력 장치 중 이름에 'respeaker'/'seeed' 가 든 첫 장치 인덱스. 없으면 None(기본 장치)."""
    import sounddevice as sd  # 지연 import

    for idx, dev in enumerate(sd.query_devices()):
        name = str(dev.get("name", "")).lower()
        if dev.get("max_input_channels", 0) >= 1 and ("respeaker" in name or "seeed" in name):
            return idx
    return None


class SilenceWatch:
    """연속 '디지털 무음' 감시 — 장치 오선택(ch0 무음) 함정을 즉시 드러낸다.

    update(samples)는 연속 chunks개 무음이 된 순간 한 번만 경고 문자열을 돌려주고,
    소리가 다시 들어오면 리셋돼 다음 무음 구간에서 또 한 번 경고한다.
    다채널 (n, C) 입력이면 ch0(처리채널) 기준. threshold 는 정상 환경소음 RMS
    보다 훨씬 낮게 잡아 '진짜 0에 가까운' 입력만 무음으로 본다.
    """

    def __init__(self, threshold: float = 1e-5, chunks: int = 5):
        self._threshold = threshold
        self._chunks = chunks
        self._run = 0
        self._warned = False

    def update(self, samples: np.ndarray) -> str | None:
        x = np.asarray(samples)
        mono = x[:, 0] if x.ndim == 2 else x
        rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64)))) if mono.size else 0.0
        if rms >= self._threshold:
            self._run, self._warned = 0, False
            return None
        self._run += 1
        if self._run >= self._chunks and not self._warned:
            self._warned = True
            return (f"[audio] 경고: 입력이 {self._run}청크 연속 무음 — 장치 오선택일 수 있음. "
                    "장치 목록 확인: python -c \"import sounddevice as sd; print(sd.query_devices())\" "
                    "→ --device N 지정")
        return None


def _resample(data: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """간단 선형 리샘플링(테스트용). 정밀 작업은 librosa.resample 권장."""
    duration = len(data) / src_sr
    dst_len = int(duration * dst_sr)
    x_old = np.linspace(0.0, duration, num=len(data), endpoint=False)
    x_new = np.linspace(0.0, duration, num=dst_len, endpoint=False)
    return np.interp(x_new, x_old, data).astype(np.float32)
