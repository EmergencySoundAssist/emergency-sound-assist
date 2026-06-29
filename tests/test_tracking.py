"""doa.tracking 단위 테스트 (순수 로직, 하드웨어 불필요).

circular_median 의 원형·강건성과 DirectionTracker 의 게이팅/다수결을 검증한다.
실행: 레포 루트에서 `pytest tests/test_tracking.py`
"""

import pytest

from doa.tracking import DirectionTracker, circular_median


def _circ_close(a, b, tol=2.0):
    return abs((a - b + 180.0) % 360.0 - 180.0) <= tol


# ---------------------------------------------------------------------------
# circular_median
# ---------------------------------------------------------------------------

def test_empty_returns_none():
    assert circular_median([]) is None


def test_single_value():
    assert circular_median([137.0]) == pytest.approx(137.0)


def test_cluster_mean_like():
    assert _circ_close(circular_median([10, 12, 8, 11]), 10.5)


def test_wraparound_cluster_near_zero():
    """359°/1° 처럼 0° 경계에 걸친 묶음도 0° 부근으로 모인다 (선형 평균의 180° 함정 회피)."""
    assert _circ_close(circular_median([359, 1, 0, 358, 2]), 0.0)


def test_outvotes_single_180_flip():
    """다수(정면)에 ±180° 로 튄 한 프레임 → 중앙값은 다수 쪽으로 (반사 아웃보팅)."""
    assert _circ_close(circular_median([10, 12, 8, 190, 11]), 10.5)


def test_normalizes_out_of_range():
    assert circular_median([365.0]) == pytest.approx(5.0)
    assert _circ_close(circular_median([370, 368, 366]), 8.0)  # = 368 % 360


# ---------------------------------------------------------------------------
# DirectionTracker — 게이팅 + 다수결
# ---------------------------------------------------------------------------

def test_below_conf_min_is_uncertain():
    """신뢰도 미만 프레임만 들어오면 방향 확정 안 함(None)."""
    t = DirectionTracker(maxlen=5, conf_min=6.0, min_frames=2)
    r = t.update(90.0, conf=3.0)
    assert r.angle is None
    assert r.n_confident == 0


def test_needs_min_frames():
    """신뢰 프레임이 min_frames 미만이면 아직 불확실, 충족하면 확정."""
    t = DirectionTracker(maxlen=5, conf_min=6.0, min_frames=2)
    assert t.update(90.0, conf=10.0).angle is None        # 1개 — 부족
    assert _circ_close(t.update(92.0, conf=10.0).angle, 91.0)  # 2개 — 확정


def test_flip_frame_outvoted_in_stream():
    """안정 스트림에 반대편 한 프레임이 끼어도 대표 방향은 유지."""
    t = DirectionTracker(maxlen=5, conf_min=6.0, min_frames=2)
    for a in (90, 91, 89):
        t.update(a, conf=10.0)
    r = t.update(270.0, conf=10.0)   # 반대편으로 튐
    assert _circ_close(r.angle, 90.0)


def test_low_conf_frame_excluded_from_vote():
    """저신뢰로 튄 프레임은 투표에서 빠져 대표 방향에 영향 없음."""
    t = DirectionTracker(maxlen=5, conf_min=6.0, min_frames=2)
    t.update(90.0, conf=10.0)
    t.update(91.0, conf=10.0)
    r = t.update(270.0, conf=2.0)    # 반대편이지만 저신뢰 → 무시
    assert _circ_close(r.angle, 90.5)
    assert r.n_confident == 2


def test_window_ages_out_old_frames():
    """창(maxlen)을 벗어난 옛 프레임은 빠지고, 무음(None)도 자연 노후화."""
    t = DirectionTracker(maxlen=3, conf_min=6.0, min_frames=2)
    t.update(90.0, conf=10.0)
    t.update(90.0, conf=10.0)
    t.update(None, conf=0.0)         # 무음 (창=3: 90,90,None → 아직 90 둘)
    t.update(None, conf=0.0)         # 창=3: 90,None,None → 90 하나
    r = t.update(None, conf=0.0)     # 창=3: None,None,None → 90 모두 밖
    assert r.angle is None
    assert r.n_confident == 0
