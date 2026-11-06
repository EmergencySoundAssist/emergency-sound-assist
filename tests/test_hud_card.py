"""HUD LED 카드 순수 로직 테스트 — 방향→위치, 음압 강도→미터/글로우/퍼짐, 밝기 감쇠."""

import pytest

from core.types import Direction, Motion










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




def _view(direction, angle_deg=None, subtitle="", sound="구급차"):
    from hud.viewmodel import HudView
    from core.types import Motion
    return HudView(
        emergency=True, sound_text=sound, direction=direction,
        direction_text="좌측", motion_text="접근 중", subtitle=subtitle,
        confidence=0.9, angle_deg=angle_deg,
        motion=Motion.APPROACHING,
    )


def test_render_current_state_no_crash():
    """angle_deg 없음(이산 방향 폴백): 긴급 카드가 크래시 없이 그려진다."""
    import pygame
    from core.types import Direction
    r, surf = _renderer_and_surface()
    r._draw_emergency(surf, _view(Direction.LEFT, subtitle="길 터주세요"), 1280, 720)
    assert r._frame == 1                              # 신규 카드 렌더 경로 확인(red-green)
    assert pygame.surfarray.array3d(surf).sum() > 0   # 뭔가 그려짐


def test_render_future_state_no_crash():
    """angle_deg 있음(DoA 연속 각도): 긴급 카드가 크래시 없이 렌더."""
    import pygame
    from core.types import Direction
    r, surf = _renderer_and_surface()
    r._draw_emergency(surf, _view(Direction.RIGHT, angle_deg=60.0,
                                  sound="소방차"), 1280, 720)
    assert r._frame == 1
    assert pygame.surfarray.array3d(surf).sum() > 0




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











# ── 퍼지는 깜빡임(ripple) ───────────────────────────────────────────────









# ---------------------------------------------------------------------------
# 음압 → 시각량 (v3). 미터·글로우·퍼짐이 전부 이 하나에서 나온다.
# ---------------------------------------------------------------------------
from hud.card import SPL_RANGE, DBFS_RANGE, spl_intensity


def test_spl_intensity_spans_the_calibrated_range():
    assert spl_intensity(SPL_RANGE[0], calibrated=True) == 0.0
    assert spl_intensity(SPL_RANGE[1], calibrated=True) == 1.0
    mid = spl_intensity((SPL_RANGE[0] + SPL_RANGE[1]) / 2, calibrated=True)
    assert 0.45 < mid < 0.55


def test_spl_intensity_uses_a_different_scale_when_uncalibrated():
    """dBFS 는 항상 음수다. dB SPL 눈금을 쓰면 미터가 늘 0 에 붙는다."""
    assert spl_intensity(-4.0, calibrated=False) > 0.8      # 풀스케일 근처 = 큼
    assert spl_intensity(-4.0, calibrated=True) == 0.0      # SPL 눈금에선 범위 밖
    assert spl_intensity(DBFS_RANGE[0], calibrated=False) == 0.0
    assert spl_intensity(DBFS_RANGE[1], calibrated=False) == 1.0


def test_spl_intensity_clamps_and_handles_missing():
    assert spl_intensity(None, calibrated=True) == 0.0
    assert spl_intensity(999.0, calibrated=True) == 1.0
    assert spl_intensity(-999.0, calibrated=True) == 0.0








from hud.card import Layout








SUITE_SIZES = [(1280, 360), (1280, 720), (640, 180), (1920, 540), (800, 480), (320, 180)]






# ---------------------------------------------------------------------------
# 레이더 (v4) — 방향은 아크, 음압은 채워지는 링 수
# ---------------------------------------------------------------------------
from hud.card import RINGS, ARC_HALF_DEG, ring_levels, arc_center_deg, arc_bounds


def test_ring_levels_fill_from_inside_out():
    """안쪽 링이 먼저 찬다 — 바깥이 켜졌는데 안쪽이 꺼져 있으면 안 된다."""
    for t in (0.1, 0.3, 0.5, 0.7, 0.9, 1.0):
        lv = ring_levels(t)
        lit = [i for i, v in enumerate(lv) if v > 0]
        assert lit == list(range(len(lit))), f"t={t}: 구멍이 뚫렸다 {lv}"


def test_ring_levels_bounds():
    assert ring_levels(0.0) == [0.0] * RINGS
    assert ring_levels(1.0) == [1.0] * RINGS
    assert all(0.0 <= v <= 1.0 for t in (0.13, 0.47, 0.82) for v in ring_levels(t))


def test_quiet_sound_still_lights_one_ring():
    """아주 작아도 '감지됐다'는 보여야 한다 — 0칸이면 미탐지와 구분이 안 된다."""
    assert ring_levels(0.01)[0] > 0


def test_partial_ring_separates_levels_that_would_collide_on_whole_counts():
    """79dB 와 88dB 는 둘 다 3칸이지만 마지막 칸 밝기로 갈려야 한다.

    9dB 차이는 소리 세기로 약 8배다. 화면이 같으면 안 된다.
    """
    a, b = ring_levels(0.44), ring_levels(0.60)          # 79dB, 88dB
    assert sum(1 for v in a if v > 0) == sum(1 for v in b if v > 0)   # 칸 수는 같고
    assert a != b, "칸 수가 같은 두 음압이 완전히 동일하게 표시된다"
    assert a[2] < b[2], "마지막 칸 밝기가 음압을 반영하지 않는다"


def test_ring_levels_are_monotonic_in_total():
    tot = [sum(ring_levels(i / 20)) for i in range(21)]
    assert all(x <= y + 1e-9 for x, y in zip(tot, tot[1:])), "음압이 올라가는데 총량이 줄었다"


def test_every_direction_points_a_different_way():
    from core.types import Direction
    seen = {d: arc_center_deg(None, d) for d in
            (Direction.FRONT, Direction.REAR, Direction.LEFT, Direction.RIGHT)}
    assert all(v is not None for v in seen.values())
    assert len(set(seen.values())) == 4, f"두 방향이 같은 각도를 쓴다: {seen}"


def test_unknown_direction_has_no_angle():
    """모르면 한 곳을 단정하지 않는다 — 렌더러가 링 전체를 균등하게 처리한다."""
    from core.types import Direction
    assert arc_center_deg(None, Direction.UNKNOWN) is None


def test_rear_left_and_rear_right_are_distinguishable():
    """후방인데 어느 쪽인지 — 소리를 못 듣는 운전자에겐 어느 거울을 볼지의 문제다.

    4분면으로 스냅하면 둘 다 '후방'이 되어 화면이 같아진다. DoA 가 주는 연속
    각도를 그대로 써야 갈린다.
    """
    from doa.estimator import _to_vehicle_angle

    def raw_for(vehicle_deg):
        return float(min(range(360),
                         key=lambda r: abs((_to_vehicle_angle(r) - vehicle_deg + 180) % 360 - 180)))

    from core.types import Direction
    right_rear = arc_center_deg(raw_for(150.0), Direction.REAR)   # 후방 우측
    left_rear = arc_center_deg(raw_for(210.0), Direction.REAR)    # 후방 좌측
    assert right_rear != left_rear, "후방 좌/우가 같은 각도로 뭉갰다"
    # 화면 각도: 270=아래. 우후방은 270보다 크고(우하), 좌후방은 작다(좌하).
    assert right_rear > 270.0 > left_rear


def test_measured_angle_wins_over_the_quadrant_fallback():
    from core.types import Direction
    from doa.estimator import _to_vehicle_angle
    raw = float(min(range(360),
                    key=lambda r: abs((_to_vehicle_angle(r) - 150.0 + 180) % 360 - 180)))
    assert arc_center_deg(raw, Direction.REAR) != arc_center_deg(None, Direction.REAR)


def test_arc_bounds_are_centred_on_the_angle():
    a0, a1 = arc_bounds(270.0)
    assert (a0 + a1) / 2 == 270.0
    assert a1 - a0 == 2 * ARC_HALF_DEG


# ---------------------------------------------------------------------------
# Layout (레이더 기준) — 좌표가 한 곳에서만 나오는지
# ---------------------------------------------------------------------------

def test_layout_matches_the_reference_resolution():
    """1280×360 기준값. 폰트도 같은 곳에서 나온다."""
    lo = Layout.for_size(1280, 360)
    assert lo.margin == 72
    assert lo.veh_xy == (72, 78)
    assert lo.cap_cy == 322
    assert (lo.radar_cx, lo.radar_cy) == (700, 158)
    assert (lo.radar_rx, lo.radar_ry) == (150, 72)
    assert (lo.ring_gap, lo.ring_w) == (12, 6)
    assert lo.f_veh == 104
    assert lo.f_state == 44


def test_layout_is_frozen_so_no_one_can_drift_a_coordinate():
    lo = Layout.for_size(1280, 360)
    with pytest.raises(Exception):
        lo.radar_cy = 999


def test_layout_scales_to_other_sizes():
    small = Layout.for_size(640, 180)
    assert small.margin == 36
    assert small.radar_rx > 0 and small.radar_ry > 0
    assert small.ring_w >= 3 and small.ring_gap >= 5
    big = Layout.for_size(1920, 540)
    assert big.radar_rx > small.radar_rx


def test_radar_stays_inside_the_screen_and_above_the_caption():
    """레이더가 화면 밖으로 나가거나 자막 줄을 침범하면 안 된다."""
    for w, h in [(1280, 360), (640, 180), (1920, 540), (800, 480), (1280, 720)]:
        lo = Layout.for_size(w, h)
        reach_x = lo.radar_rx + lo.ring_w // 2 + (RINGS - 1) * lo.ring_gap
        reach_y = lo.radar_ry + lo.ring_w // 2 + (RINGS - 1) * lo.ring_gap
        assert lo.radar_cx - reach_x >= 0, f"{w}x{h}: 레이더가 왼쪽으로 넘침"
        assert lo.radar_cx + reach_x <= w, f"{w}x{h}: 레이더가 오른쪽으로 넘침"
        assert lo.radar_cy - reach_y >= 0, f"{w}x{h}: 레이더가 위로 넘침"
        assert lo.radar_cy + reach_y < lo.cap_cy, f"{w}x{h}: 레이더가 자막 줄을 침범"
        assert lo.db_right <= w


def test_vehicle_font_never_overruns_the_radar():
    """폰트는 h 로, 텍스트 칸은 w 로 자란다 — 상한이 없으면 세로 긴 화면에서 겹친다."""
    for w, h in [(1280, 360), (1280, 720), (800, 480), (640, 180)]:
        lo = Layout.for_size(w, h)
        left = lo.radar_cx - lo.radar_rx - lo.ring_w // 2 - (RINGS - 1) * lo.ring_gap
        assert lo.f_veh * 3.48 <= (left - lo.margin), f"{w}x{h}: 차종 글자가 레이더를 침범"
