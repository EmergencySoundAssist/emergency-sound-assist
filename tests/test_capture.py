"""audio.capture 순수 로직(장치 결정·무음 감시) 테스트 — 하드웨어/sounddevice 불필요."""

import numpy as np

from audio.capture import SilenceWatch, _find_respeaker_index, _resolve_input_device

DEVS = [
    {"name": "HDA NVidia: HDMI", "max_input_channels": 0},
    {"name": "USB Camera: Audio", "max_input_channels": 1},
    {"name": "ReSpeaker 4 Mic Array (UAC1.0)", "max_input_channels": 6},
]


def test_find_respeaker_by_name():
    assert _find_respeaker_index(DEVS) == 2


def test_find_respeaker_none_when_absent():
    assert _find_respeaker_index(DEVS[:2]) is None


def test_resolve_multichannel_prefers_respeaker():
    device, label = _resolve_input_device(None, 6, DEVS)
    assert device == 2
    assert "ReSpeaker" in label


def test_resolve_explicit_device_wins():
    device, label = _resolve_input_device(1, 6, DEVS)
    assert device == 1
    assert "USB Camera" in label


def test_resolve_mono_keeps_system_default():
    device, label = _resolve_input_device(None, 1, DEVS)
    assert device is None
    assert "기본 장치" in label


def test_resolve_multichannel_without_respeaker_falls_back_with_hint():
    device, label = _resolve_input_device(None, 6, DEVS[:2])
    assert device is None
    assert "--device" in label       # 폴백 사실 + 해결 힌트가 라벨에 드러나야 함

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
