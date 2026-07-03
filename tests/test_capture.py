"""audio.capture 순수 로직(장치 결정·무음 감시) 테스트 — 하드웨어/sounddevice 불필요."""

import numpy as np

from audio.capture import SilenceWatch

SILENT = np.zeros(16000, dtype=np.float32)
LOUD = np.full(16000, 0.1, dtype=np.float32)


def test_silence_watch_warns_after_consecutive_silence():
    w = SilenceWatch(chunks=3)
    assert w.update(SILENT) is None
    assert w.update(SILENT) is None
    assert w.update(SILENT) is not None      # 3번째 연속 무음에서 경고


def test_silence_watch_warns_once_per_episode():
    w = SilenceWatch(chunks=2)
    w.update(SILENT)
    assert w.update(SILENT) is not None
    assert w.update(SILENT) is None          # 같은 무음 구간에서 반복 경고 금지


def test_silence_watch_resets_on_sound():
    w = SilenceWatch(chunks=2)
    w.update(SILENT)
    w.update(SILENT)                          # 경고 1회 소진
    assert w.update(LOUD) is None             # 소리 → 리셋
    w.update(SILENT)
    assert w.update(SILENT) is not None       # 새 무음 구간 → 다시 경고


def test_silence_watch_multichannel_uses_ch0():
    w = SilenceWatch(chunks=1)
    x = np.zeros((16000, 6), dtype=np.float32)
    x[:, 1] = 0.5                             # ch1만 소리 — ch0은 무음
    assert w.update(x) is not None


def test_silence_watch_empty_chunk_counts_as_silence():
    w = SilenceWatch(chunks=1)
    assert w.update(np.zeros(0, dtype=np.float32)) is not None
