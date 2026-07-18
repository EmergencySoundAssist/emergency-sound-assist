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


# ── Task 3: 경적 접근 억제 (strip_lit / _subline) ─────────────────────────

def test_strip_lit_horn_always_on():
    """경적은 접근 깜빡임 없이 상시 점등 (blink 이라면 꺼졌을 프레임에도 켜짐)."""
    from hud.card import strip_lit
    from core.types import Motion
    # 비경적·접근이면 period=18 → frame=10 은 꺼짐(10 % 18 = 10 >= 9)
    assert strip_lit(False, None, Motion.APPROACHING, 10) is False
    # 경적이면 같은 상황에서도 항상 켜짐
    assert strip_lit(True, None, Motion.APPROACHING, 10) is True


def test_strip_lit_non_horn_follows_blink():
    """비경적은 기존 blink 규칙(주기 18: 앞 절반 on)."""
    from hud.card import strip_lit
    from core.types import Motion
    assert strip_lit(False, None, Motion.APPROACHING, 0) is True    # 0 < 9
    assert strip_lit(False, None, Motion.STEADY, 999) is True       # 유지=상시


def test_subline_horn_omits_motion():
    """경적: 방향만, '접근/이동' 문구 없음."""
    from hud.renderer import Renderer
    from core.types import Direction
    from hud.viewmodel import HudView
    horn_left = HudView(
        emergency=True, sound_text="경적", direction=Direction.LEFT,
        direction_text="좌측", motion_text="접근 중", subtitle="",
        confidence=0.8, is_horn=True,
    )
    assert Renderer._subline(horn_left) == "좌측"      # 방향만
    horn_unknown = HudView(
        emergency=True, sound_text="경적", direction=Direction.UNKNOWN,
        direction_text="방향 미상", motion_text="접근 중", subtitle="",
        confidence=0.8, is_horn=True,
    )
    assert Renderer._subline(horn_unknown) == ""        # 방향 미상이면 빈 문자열


# ── Task 4: 렌더러 통합 (STT 방향 바 / 점 애니메이션 / 멀티라인 자막) ──────

def _normal_view_card(subtitle=""):
    from hud.viewmodel import HudView
    from core.types import Direction, Motion
    return HudView(
        emergency=False, sound_text="일반 도로 소음", direction=Direction.UNKNOWN,
        direction_text="방향 미상", motion_text="이동 미상", subtitle=subtitle,
        confidence=0.3, motion=Motion.UNKNOWN,
    )


def test_normal_mode_draws_idle_direction_strip():
    """평상시(STT) 화면에도 방향 바가 그려진다 — 스트립 행에 픽셀이 존재."""
    import pygame
    r, surf = _renderer_and_surface()               # 1280x720
    r._draw_normal(surf, _normal_view_card(), 1280, 720)
    arr = pygame.surfarray.array3d(surf)            # (w, h, 3)
    y = int(720 * 0.42)
    strip_band = arr[:, y - 15:y + 55, :]
    assert strip_band.sum() > 0                     # idle 바가 그려짐


def test_normal_mode_dots_animate_over_frames():
    """자막이 없으면 점 애니메이션 — 프레임이 진행되며 렌더가 달라진다."""
    import pygame
    r, surf = _renderer_and_surface()
    frames = []
    for _ in range(6):
        surf.fill((0, 0, 0))
        r._draw_normal(surf, _normal_view_card(), 1280, 720)
        band = pygame.surfarray.array3d(surf)[:, 720 - 720 // 3:, :]  # 하단 밴드
        frames.append(int(band.sum()))
    assert len(set(frames)) > 1                     # 애니메이션(정지 아님)


def test_normal_mode_long_subtitle_wraps_without_crash():
    """긴 자막이 여러 줄로 렌더된다(크래시 없음)."""
    import pygame
    r, surf = _renderer_and_surface()
    long_text = "앞차가 급정거했으니 차간 거리를 충분히 확보하고 서행하세요 다시 한 번 안내드립니다"
    r._draw_normal(surf, _normal_view_card(subtitle=long_text), 1280, 720)
    assert pygame.surfarray.array3d(surf).sum() > 0


def test_emergency_horn_strip_never_dark():
    """경적: 스트립이 매 프레임 '점등'(차종색) 상태 — 접근 깜빡임으로 소등되지 않음.

    깜빡임 OFF 세그먼트는 DIM(회색)으로 그려지므로, 단순 픽셀 유무가 아니라
    '차종색(주황) 점등 세그먼트'가 매 프레임 존재하는지로 확인한다.
    """
    import numpy as np
    import pygame
    from hud.viewmodel import HudView
    from hud.renderer import VEH_OTHER
    from core.types import Direction, Motion
    r, surf = _renderer_and_surface()
    horn = HudView(
        emergency=True, sound_text="경적", direction=Direction.LEFT,
        direction_text="좌측", motion_text="접근 중", subtitle="",
        confidence=0.8, motion=Motion.APPROACHING, is_horn=True,
    )
    target = np.array(VEH_OTHER)
    lit_counts = []
    for _ in range(20):                             # 한 blink 주기(18) 이상
        surf.fill((0, 0, 0))
        r._draw_emergency(surf, horn, 1280, 720)
        # 스트립 행(중앙 y=h//2)만 검사 — 헤더의 주황 강조선/서브라인 제외
        band = pygame.surfarray.array3d(surf).astype(int)[:, 355:410, :]
        lit = (np.abs(band - target).sum(axis=2) < 80)   # 차종색 점등 세그먼트
        lit_counts.append(int(lit.sum()))
    assert all(c > 0 for c in lit_counts)           # 어떤 프레임에도 소등 안 됨


# ── Task: 방향 바 각도 매핑 (azimuth_to_bar_index / direction_visible) ──────

def test_azimuth_bar_hides_front():
    """raw 90° → 차량 전방(0°) → 숨김(None)."""
    from hud.card import azimuth_to_bar_index
    assert azimuth_to_bar_index(90) is None


def test_azimuth_bar_left_center_right():
    """차량 우→오른쪽끝, 후→중앙, 좌→왼쪽끝 (config: REAR_RAW_DEG=270, MIRROR=True)."""
    from hud.card import azimuth_to_bar_index
    assert azimuth_to_bar_index(0) == 14     # raw0 → 차량 우(90°)
    assert azimuth_to_bar_index(270) == 7    # raw270 → 차량 후(180°)
    assert azimuth_to_bar_index(180) == 0    # raw180 → 차량 좌(270°)


def test_direction_visible_hides_front_only():
    from hud.card import direction_visible
    from core.types import Direction
    assert direction_visible(Direction.FRONT) is False
    assert direction_visible(Direction.LEFT) is True
    assert direction_visible(Direction.REAR) is True
    assert direction_visible(Direction.RIGHT) is True
    assert direction_visible(Direction.UNKNOWN) is True


# ── 접근 빠르기 → 퍼짐 반경 / 퍼지는 깜빡임 ───────────────────────────────

def test_spread_for_speed_widens_with_speed():
    from hud.card import spread_for_speed
    assert spread_for_speed(None) == 3        # 접근 아님 → 기본
    assert spread_for_speed(1) == 2           # 느린 접근 → 좁게
    assert spread_for_speed(5) == 6           # 빠른 접근 → 넓게
    assert spread_for_speed(1) < spread_for_speed(5)
    assert spread_for_speed(99) == 6          # 범위 밖 클램프


def test_should_ripple_only_when_approaching_and_not_horn():
    from hud.card import should_ripple
    from core.types import Motion
    assert should_ripple(False, Motion.APPROACHING) is True
    assert should_ripple(True, Motion.APPROACHING) is False    # 경적 = 상시
    assert should_ripple(False, Motion.STEADY) is False
    assert should_ripple(False, Motion.RECEDING) is False


def test_ripple_brightness_expands_and_resets():
    from hud.card import ripple_brightness, RIPPLE_PERIOD
    # 위상 0: 중앙만 밝고(=1) 바깥은 꺼짐
    assert ripple_brightness(7, 7, 6, 0) == 1.0
    assert ripple_brightness(10, 7, 6, 0) == 0.0
    # 위상이 진행되면 바깥 세그먼트가 켜진다(퍼짐)
    mid = ripple_brightness(10, 7, 6, RIPPLE_PERIOD // 2)
    assert mid > 0.0
    # 주기마다 리셋(동일 위상 반복)
    assert ripple_brightness(9, 7, 6, 3) == ripple_brightness(9, 7, 6, 3 + RIPPLE_PERIOD)


def test_emergency_front_hides_bar():
    """전방(각도) → 방향 바 미표시: 스트립 행에 차종색 세그먼트가 없다."""
    import numpy as np
    import pygame
    from hud.viewmodel import HudView
    from hud.renderer import VEH_AMBULANCE
    from core.types import Direction, Motion
    r, surf = _renderer_and_surface()
    surf.fill((0, 0, 0))
    front = HudView(emergency=True, sound_text="구급차", direction=Direction.FRONT,
                    direction_text="전방", motion_text="접근 중", subtitle="",
                    confidence=0.9, angle_deg=90.0, motion=Motion.APPROACHING)  # raw90→차량 전방
    r._draw_emergency(surf, front, 1280, 720)
    band = pygame.surfarray.array3d(surf).astype(int)[:, 340:420, :]   # 스트립 행만
    lit = (np.abs(band - np.array(VEH_AMBULANCE)).sum(axis=2) < 80)
    assert lit.sum() == 0                                              # 바 숨김


def test_emergency_angle_positions_bar_left_vs_right():
    """각도에 따라 점등 클러스터가 좌/우로 이동(실시간 각도 구동)."""
    import numpy as np
    import pygame
    from hud.viewmodel import HudView
    from hud.renderer import VEH_AMBULANCE
    from core.types import Direction, Motion

    def cluster_cx(raw):
        r, surf = _renderer_and_surface()
        surf.fill((0, 0, 0))
        v = HudView(emergency=True, sound_text="구급차", direction=Direction.LEFT,
                    direction_text="좌측", motion_text="접근 중", subtitle="",
                    confidence=0.9, angle_deg=raw, motion=Motion.APPROACHING)
        r._draw_emergency(surf, v, 1280, 720)
        arr = pygame.surfarray.array3d(surf)[:, 340:420, :]           # 스트립 행만
        mask = (np.abs(arr.astype(int) - np.array(VEH_AMBULANCE)).sum(axis=2) < 60)
        xs = np.where(mask.any(axis=1))[0]
        return xs.mean() if xs.size else None

    left_cx = cluster_cx(180)     # raw180 → 차량 좌 → 왼쪽
    right_cx = cluster_cx(0)      # raw0   → 차량 우 → 오른쪽
    assert left_cx is not None and right_cx is not None
    assert left_cx < right_cx
