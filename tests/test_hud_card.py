"""HUD LED 카드 순수 로직 테스트 — 방향→위치, 빠르기→blink, 밝기 감쇠."""

from core.types import Direction, Motion
from hud.card import (
    direction_to_index, blink_spec, is_lit_now, segment_brightness,
)


def test_angle_maps_center_left_right():
    """각도: 0°→중앙, -90°→맨왼쪽, +90°→맨오른쪽 (n=15)."""
    assert direction_to_index(0.0, Direction.UNKNOWN, n=15) == 7
    assert direction_to_index(-90.0, Direction.UNKNOWN, n=15) == 0
    assert direction_to_index(90.0, Direction.UNKNOWN, n=15) == 14


def test_angle_clamps_out_of_range():
    """범위 밖 각도는 끝으로 고정."""
    assert direction_to_index(-200.0, Direction.UNKNOWN, n=15) == 0
    assert direction_to_index(200.0, Direction.UNKNOWN, n=15) == 14


def test_discrete_fallback_when_no_angle():
    """angle None 이면 이산 방향으로 구역 스냅."""
    assert direction_to_index(None, Direction.LEFT, n=15) < 7    # 왼쪽
    assert direction_to_index(None, Direction.RIGHT, n=15) > 7   # 오른쪽
    assert direction_to_index(None, Direction.FRONT, n=15) == 7  # 중앙
    assert direction_to_index(None, Direction.UNKNOWN, n=15) == 7


def test_blink_from_speed_level():
    """speed_level 1~5 → 느림(30)~빠름(9) 주기."""
    assert blink_spec(5, Motion.APPROACHING)[1] == 9    # 빠르게
    assert blink_spec(1, Motion.APPROACHING)[1] == 30   # 느리게
    assert blink_spec(5, Motion.APPROACHING)[0] == "빠르게"


def test_blink_fallback_to_motion_when_no_speed():
    """speed_level None → Motion 폴백: 접근=보통 blink, 그외=상시(주기0)."""
    assert blink_spec(None, Motion.APPROACHING) == ("접근 중", 18)
    assert blink_spec(None, Motion.RECEDING) == ("멀어짐", 0)
    assert blink_spec(None, Motion.STEADY) == ("유지", 0)


def test_is_lit_now_steady_always_on():
    """주기 0(상시)은 항상 점등."""
    assert is_lit_now(0, 0) is True
    assert is_lit_now(0, 999) is True


def test_is_lit_now_blinks():
    """주기 9면 앞 절반 on, 뒤 절반 off."""
    assert is_lit_now(9, 0) is True
    assert is_lit_now(9, 4) is True
    assert is_lit_now(9, 5) is False


def test_segment_brightness_decays():
    """중심이 가장 밝고(1.0) 반경 밖은 0."""
    assert segment_brightness(7, 7, radius=2) == 1.0
    assert segment_brightness(9, 7, radius=2) == 0.0   # 2칸 밖 경계(>radius)
    assert 0.0 < segment_brightness(8, 7, radius=2) < 1.0
