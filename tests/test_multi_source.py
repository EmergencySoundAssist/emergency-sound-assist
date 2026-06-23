"""doa.multi_source 단위 테스트 (순수 로직만, 하드웨어·pyroomacoustics 불필요).

pick_peaks / mic_locations 는 numpy/scipy 만 쓰므로 합성 스펙트럼으로 검증한다.
SRP-PHAT 본체(spatial_spectrum)는 pyroomacoustics 의존이라 여기서 다루지 않는다.
실행: 레포 루트에서 `pytest tests/test_multi_source.py`
"""

import numpy as np
import pytest

from doa.multi_source import mic_locations, pick_peaks

AZ = np.arange(360.0)  # 1° 간격 방위각 그리드


def _gaussian_peak(center_deg, width_deg=5.0, height=1.0, n=360):
    """원형 그리드 위 가우시안 봉우리 (0°=360° 연결 고려)."""
    d = np.abs(AZ - center_deg)
    d = np.minimum(d, n - d)  # 원형 최소 거리
    return height * np.exp(-(d ** 2) / (2 * width_deg ** 2))


# ---------------------------------------------------------------------------
# mic_locations
# ---------------------------------------------------------------------------

def test_mic_locations_shape_and_radius():
    locs = mic_locations(radius_m=0.05)
    assert locs.shape == (2, 4)
    # 모든 마이크가 같은 반지름 원 위
    radii = np.hypot(locs[0], locs[1])
    assert np.allclose(radii, 0.05)


# ---------------------------------------------------------------------------
# pick_peaks — 핵심 다중 추출 로직
# ---------------------------------------------------------------------------

def test_single_peak():
    spec = _gaussian_peak(90)
    peaks = pick_peaks(spec, AZ, max_src=2)
    assert len(peaks) == 1
    assert peaks[0] == pytest.approx(90, abs=2)


def test_two_well_separated_peaks():
    spec = _gaussian_peak(30) + _gaussian_peak(200)
    peaks = sorted(pick_peaks(spec, AZ, max_src=2))
    assert len(peaks) == 2
    assert peaks[0] == pytest.approx(30, abs=2)
    assert peaks[1] == pytest.approx(200, abs=2)


def test_strongest_first():
    """에너지 큰 봉우리가 먼저 나온다."""
    spec = _gaussian_peak(30, height=0.6) + _gaussian_peak(200, height=1.0)
    peaks = pick_peaks(spec, AZ, max_src=2)
    assert peaks[0] == pytest.approx(200, abs=2)  # 더 센 쪽이 1순위


def test_max_src_caps_result():
    spec = _gaussian_peak(0) + _gaussian_peak(120) + _gaussian_peak(240)
    assert len(pick_peaks(spec, AZ, max_src=2)) == 2
    assert len(pick_peaks(spec, AZ, max_src=3)) == 3


def test_min_sep_merges_close_peaks():
    """min_sep_deg 보다 가까운 두 봉우리는 하나로 본다."""
    spec = _gaussian_peak(100) + _gaussian_peak(115)  # 15° 간격
    peaks = pick_peaks(spec, AZ, max_src=2, min_sep_deg=30.0)
    assert len(peaks) == 1


def test_height_ratio_filters_weak_peak():
    """약한 봉우리는 height_ratio 임계로 제거."""
    spec = _gaussian_peak(90, height=1.0) + _gaussian_peak(270, height=0.3)
    peaks = pick_peaks(spec, AZ, max_src=2, height_ratio=0.5)
    assert len(peaks) == 1
    assert peaks[0] == pytest.approx(90, abs=2)


def test_wraparound_peak_near_zero():
    """0°/360° 경계에 걸친 봉우리도 하나로 잡힌다 (이중 검출 안 됨)."""
    spec = _gaussian_peak(0)  # 359°와 1° 양쪽으로 퍼짐
    peaks = pick_peaks(spec, AZ, max_src=2)
    assert len(peaks) == 1
    assert peaks[0] == pytest.approx(0, abs=2) or peaks[0] == pytest.approx(359, abs=2)


def test_flat_or_empty_spectrum_returns_empty():
    assert pick_peaks(np.zeros(360), AZ) == []
    assert pick_peaks(np.array([]), np.array([])) == []
