"""augment 순수 함수 테스트 — SNR 정확도·확성기 대역제한."""
import numpy as np
import pytest

from finetune.augment import mix_at_snr, simulate_loudspeaker

SR = 16000


def _rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def _tone(freq, seconds=1.0, amp=0.1):
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_mix_at_snr_achieves_target_snr():
    rng = np.random.default_rng(0)
    speech = _tone(440, amp=0.1)
    noise = (0.1 * rng.standard_normal(len(speech))).astype(np.float32)
    for target in (10.0, 5.0, 0.0):
        mixed = mix_at_snr(speech, noise, target)
        added = mixed - speech          # 피크 정규화가 안 일어난 진폭이므로 성립
        got = 20 * np.log10(_rms(speech) / _rms(added))
        assert abs(got - target) < 0.1, (target, got)


def test_mix_output_shape_and_dtype():
    speech = _tone(440)
    noise = _tone(100, seconds=2.0)     # 더 길어도 speech 길이에 맞춰 잘림
    mixed = mix_at_snr(speech, noise, 5.0)
    assert mixed.dtype == np.float32 and len(mixed) == len(speech)


def test_mix_rejects_short_noise():
    with pytest.raises(ValueError):
        mix_at_snr(_tone(440, seconds=1.0), _tone(100, seconds=0.5), 5.0)


def test_mix_never_clips():
    rng = np.random.default_rng(1)
    speech = _tone(440, amp=0.9)
    noise = (0.9 * rng.standard_normal(len(speech))).astype(np.float32)
    mixed = mix_at_snr(speech, noise, 0.0)
    assert np.max(np.abs(mixed)) <= 1.0 + 1e-6


def test_loudspeaker_bandlimits():
    # 소프트클립이 소신호를 ~3배 증폭하므로 절대 RMS 가 아니라
    # "대역 밖이 대역 안 대비 얼마나 죽는지" 상대 비교로 검증한다.
    ref = _rms(simulate_loudspeaker(_tone(1000, amp=0.3), SR))   # 통과대역 기준
    for freq in (100, 6000):                                     # 대역 밖
        y = simulate_loudspeaker(_tone(freq, amp=0.3), SR)
        assert y.dtype == np.float32 and len(y) == SR
        assert np.max(np.abs(y)) <= 1.0 + 1e-6
        assert _rms(y) < 0.2 * ref, (freq, _rms(y), ref)


def test_loudspeaker_shape_and_peak():
    x = _tone(1000, amp=0.9)
    y = simulate_loudspeaker(x, SR)
    assert y.dtype == np.float32 and len(y) == len(x)
    assert np.max(np.abs(y)) <= 1.0 + 1e-6
