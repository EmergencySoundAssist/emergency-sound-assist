"""오디오 로드·리샘플·크롭 유틸 테스트 (tmp 파일 사용, 네트워크 없음)."""
import numpy as np
import soundfile as sf

from finetune.audio_io import load_mono_16k, save_wav_16k, crop_or_tile


def _tone(sr, seconds=1.0, freq=440.0):
    t = np.arange(int(sr * seconds)) / sr
    return (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_load_resamples_to_16k(tmp_path):
    p = tmp_path / "a24k.wav"
    sf.write(p, _tone(24000), 24000)          # 24kHz 입력 (edge-tts mp3 와 같은 sr)
    x = load_mono_16k(p)
    assert x.dtype == np.float32
    assert abs(len(x) - 16000) <= 16          # 1초 → 16000 샘플 (±리샘플 오차)


def test_load_downmixes_stereo(tmp_path):
    p = tmp_path / "st.wav"
    stereo = np.stack([_tone(16000), _tone(16000)], axis=1)
    sf.write(p, stereo, 16000)
    x = load_mono_16k(p)
    assert x.ndim == 1 and len(x) == 16000


def test_save_roundtrip(tmp_path):
    p = tmp_path / "out.wav"
    save_wav_16k(p, _tone(16000))
    data, sr = sf.read(p, dtype="float32")
    assert sr == 16000 and data.ndim == 1 and len(data) == 16000


def test_crop_or_tile_deterministic():
    x = np.arange(100, dtype=np.float32)
    a = crop_or_tile(x, 50, np.random.default_rng(7))
    b = crop_or_tile(x, 50, np.random.default_rng(7))
    assert len(a) == 50 and np.array_equal(a, b)


def test_crop_or_tile_extends_short_input():
    x = np.arange(10, dtype=np.float32)
    y = crop_or_tile(x, 35, np.random.default_rng(0))
    assert len(y) == 35
