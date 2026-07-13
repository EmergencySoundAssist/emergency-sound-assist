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


def test_vehicle_color_mapping():
    """차종 텍스트 → LED 색. 미상/경적 등은 기타색."""
    from hud.renderer import (
        vehicle_color, VEH_AMBULANCE, VEH_POLICE, VEH_FIRE, VEH_OTHER,
    )
    assert vehicle_color("구급차") == VEH_AMBULANCE
    assert vehicle_color("경찰차") == VEH_POLICE
    assert vehicle_color("소방차") == VEH_FIRE
    assert vehicle_color("경적") == VEH_OTHER
    assert vehicle_color("사이렌") == VEH_OTHER      # 미상 계열 → 기타색


def test_bundled_font_present_and_loads():
    """Pretendard 번들 폰트가 repo에 있고 최우선 후보로 로드된다."""
    import os
    import pygame
    from hud.renderer import _FONT_CANDIDATES, load_font
    assert _FONT_CANDIDATES[0].endswith("Pretendard-SemiBold.otf")
    assert os.path.exists(_FONT_CANDIDATES[0])       # repo에 번들됨(누락 시 실패)
    pygame.init()
    f = load_font(40)
    assert f.render("긴급", True, (255, 255, 255)).get_width() > 0


def _renderer_and_surface():
    import pygame
    from hud.renderer import Renderer
    from hud.config import HudConfig
    pygame.init()
    r = Renderer(HudConfig(width=1280, height=720, fullscreen=False))
    return r, pygame.Surface((1280, 720))


def _view(direction, angle_deg=None, speed_level=None, subtitle="", sound="구급차"):
    from hud.viewmodel import HudView
    from core.types import Motion
    return HudView(
        emergency=True, sound_text=sound, direction=direction,
        direction_text="좌측", motion_text="접근 중", subtitle=subtitle,
        confidence=0.9, angle_deg=angle_deg, speed_level=speed_level,
        motion=Motion.APPROACHING,
    )


def test_render_current_state_no_crash():
    """현재 상태(angle/speed None): 성능저하 카드가 크래시 없이 그려진다."""
    import pygame
    from core.types import Direction
    r, surf = _renderer_and_surface()
    r._draw_emergency(surf, _view(Direction.LEFT, subtitle="길 터주세요"), 1280, 720)
    assert r._frame == 1                              # 신규 카드 렌더 경로 확인(red-green)
    assert pygame.surfarray.array3d(surf).sum() > 0   # 뭔가 그려짐


def test_render_future_state_no_crash():
    """미래 상태(angle·speed 채움): 완전 카드가 크래시 없이 렌더."""
    import pygame
    from core.types import Direction
    r, surf = _renderer_and_surface()
    r._draw_emergency(surf, _view(Direction.RIGHT, angle_deg=60.0, speed_level=5,
                                  sound="소방차"), 1280, 720)
    assert r._frame == 1
    assert pygame.surfarray.array3d(surf).sum() > 0


def test_led_cluster_follows_direction():
    """방향에 따라 점등 클러스터의 가로 위치가 좌/우로 갈린다."""
    import numpy as np
    import pygame
    from core.types import Direction
    from hud.renderer import VEH_AMBULANCE

    def lit_x_center(direction):
        r, surf = _renderer_and_surface()
        r._draw_emergency(surf, _view(direction), 1280, 720)
        arr = pygame.surfarray.array3d(surf)          # shape (w, h, 3)
        # 차종색(초록)에 가까운 픽셀들의 x 무게중심
        target = np.array(VEH_AMBULANCE)
        mask = (np.abs(arr.astype(int) - target).sum(axis=2) < 60)
        xs = np.where(mask.any(axis=1))[0]
        return xs.mean() if xs.size else None

    left_cx = lit_x_center(Direction.LEFT)
    right_cx = lit_x_center(Direction.RIGHT)
    assert left_cx is not None and right_cx is not None
    assert left_cx < right_cx                          # 좌측 클러스터가 더 왼쪽


# ── Task 1: wrap_text (자막 자동 줄바꿈) ──────────────────────────────────

def test_wrap_text_single_line_when_short():
    """폭에 맞으면 한 줄 그대로."""
    from hud.card import wrap_text
    lines = wrap_text("비켜", lambda s: len(s), max_width=10, max_lines=2)
    assert lines == ["비켜"]


def test_wrap_text_breaks_into_multiple_lines():
    """폭 초과 시 글자 단위로 줄바꿈 (measure=글자수, 폭=3)."""
    from hud.card import wrap_text
    lines = wrap_text("가나다라마바", lambda s: len(s), max_width=3, max_lines=2)
    assert lines == ["가나다", "라마바"]


def test_wrap_text_ellipsizes_on_overflow():
    """max_lines 초과분은 마지막 줄을 '…'로 절단."""
    from hud.card import wrap_text
    lines = wrap_text("가나다라마바사아", lambda s: len(s), max_width=3, max_lines=2)
    assert len(lines) == 2
    assert lines[0] == "가나다"
    assert lines[1].endswith("…")
    assert len(lines[1]) <= 3          # '…' 포함 폭 이내


def test_wrap_text_empty_is_empty_list():
    from hud.card import wrap_text
    assert wrap_text("", lambda s: len(s), max_width=10) == []


# ── Task 2: dots_brightness (STT 변환 중 점 3개 애니메이션) ────────────────

def test_dots_brightness_shape_and_range():
    """점 3개, 밝기 0.2~1.0 범위."""
    from hud.card import dots_brightness
    bs = dots_brightness(0)
    assert len(bs) == 3
    assert all(0.2 <= b <= 1.0 for b in bs)


def test_dots_brightness_animates():
    """프레임에 따라 값이 변한다(정지 아님) — 한 주기 안에서 서로 다른 상태 존재."""
    from hud.card import dots_brightness
    states = {tuple(round(b, 3) for b in dots_brightness(f)) for f in range(24)}
    assert len(states) > 1


def test_dots_brightness_is_periodic():
    """period 만큼 지나면 동일 상태로 되돌아온다."""
    from hud.card import dots_brightness
    assert dots_brightness(3) == dots_brightness(3 + 24)
