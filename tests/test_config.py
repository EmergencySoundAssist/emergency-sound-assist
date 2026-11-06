"""doa.config 중앙 설정이 각 모듈에 실제로 연결돼 있는지 검증.

config.py 값을 바꿨을 때 estimator/multi_source/led_ring 이 그대로 따라가는지
(상수가 config 에서 온 것인지) 확인한다.
"""

from doa import config, estimator, led_ring, multi_source


def test_estimator_calibration_from_config():
    assert estimator.REAR_RAW_DEG == config.REAR_RAW_DEG
    assert estimator.MIRROR == config.MIRROR


def test_multi_source_geometry_from_config():
    assert multi_source.MIC_RADIUS_M == config.MIC_RADIUS_M
    assert multi_source.MIC_ANGLES_DEG == config.MIC_ANGLES_DEG
    assert multi_source.FS_DEFAULT == config.FS
    assert multi_source.NFFT_DEFAULT == config.NFFT
    assert multi_source.SIREN_BAND_HZ == config.SIREN_BAND_HZ


def test_led_offset_from_config():
    assert led_ring.LED_OFFSET == config.LED_OFFSET


def test_config_has_expected_keys():
    """주요 튜닝/보정 키가 빠지지 않았는지."""
    for key in ("WINDOW", "HOP", "HOLD", "NUM_SRC", "ALGO", "HEIGHT_RATIO",
                "MIN_SEP_DEG", "THRESHOLD", "LED", "REAR_RAW_DEG", "MIRROR",
                "MIC_RADIUS_M", "LED_OFFSET", "LED_BRIGHTNESS",
                "SMOOTH", "CONF_MIN", "SMOOTH_WIN", "SMOOTH_MIN_FRAMES"):
        assert hasattr(config, key), f"config.{key} 누락"
