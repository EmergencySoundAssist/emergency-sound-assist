"""Renderer 헤드리스 렌더 스모크 — SDL dummy, 예외 없이 그려지면 통과."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pytest

pygame = pytest.importorskip("pygame")

from core.types import Direction
from hud.config import HudConfig
from hud.renderer import Renderer
from hud.viewmodel import HudView


def _emergency_view():
    return HudView(emergency=True, sound_text="구급차", direction=Direction.REAR,
                   direction_text="후방", motion_text="접근 중", subtitle="", confidence=0.9)


def _normal_view():
    return HudView(emergency=False, sound_text="일반 도로 소음", direction=Direction.UNKNOWN,
                   direction_text="방향 미상", motion_text="이동 미상",
                   subtitle="비켜주세요", confidence=0.4)


def test_renderer_draws_all_states_without_error():
    pygame.init()
    surf = pygame.Surface((320, 180))
    r = Renderer(HudConfig(width=320, height=180))
    r.draw(surf, None)              # 대기
    r.draw(surf, _emergency_view()) # 긴급
    r.draw(surf, _normal_view())    # 평상시
    pygame.quit()


# ---------------------------------------------------------------------------
# v3 고정 그리드 — 상태가 바뀌어도 좌표가 움직이지 않는다
# ---------------------------------------------------------------------------
from hud.card import Layout
from core.types import Motion


def _view(**kw):
    base = dict(
        emergency=True, sound_text="구급차", direction=Direction.RIGHT,
        direction_text="우측", motion_text="접근 중", subtitle="",
        confidence=0.9, angle_deg=None, motion=Motion.APPROACHING,
        is_horn=False, level_db=97.0, level_text="97", spl_calibrated=True,
    )
    base.update(kw)
    return HudView(**base)


def _render(view, frames=1):
    pygame.init()
    cfg = HudConfig(fullscreen=False)
    surf = pygame.Surface((cfg.width, cfg.height))
    r = Renderer(cfg)
    for _ in range(frames):
        r.draw(surf, view)
    return surf






def test_uncalibrated_view_draws_no_number():
    """미보정이면 숫자를 아예 그리지 않는다 — 단위를 속이지 않는다."""
    lo = Layout.for_size(1280, 360)
    surf = _render(_view(level_text=None, spl_calibrated=False))
    band = [surf.get_at((x, y))[:3]
            for y in range(lo.radar_cy, min(360, lo.radar_cy + 34))
            for x in range(lo.db_right - 140, lo.db_right)]
    assert all(sum(c) < 60 for c in band), "미보정인데 숫자가 그려졌다"


def test_rear_and_unknown_do_not_render_identically():
    """후방과 방향 미상이 같은 화면이면 HUD 가 모르는 방향을 단정하는 것이다.

    이전 가로 바 구현은 REAR·UNKNOWN 을 둘 다 중앙 칸으로 보내서 픽셀이 같았다 —
    영원히 구분되지 않는다. DoA 하드웨어가 없으면 UNKNOWN 이 오히려 정상 상태라,
    문구가 방향을 말해 주지 않으면 소리를 못 듣는 운전자는 뒤를 볼 이유가 없는데도
    뒤를 본다. 픽셀이 달라야 한다.
    """
    rear = _render(_view(direction=Direction.REAR, direction_text="후방"))
    unknown = _render(_view(direction=Direction.UNKNOWN, direction_text="방향 미상"))
    lo = Layout.for_size(1280, 360)
    rows = range(lo.state_xy[1], min(360, lo.state_xy[1] + lo.f_state))
    diff = [(x, y) for y in rows for x in range(lo.margin, (lo.radar_cx - lo.radar_rx))
            if rear.get_at((x, y))[:3] != unknown.get_at((x, y))[:3]]
    assert diff, "후방과 방향 미상이 픽셀까지 동일하다 — 모르는 방향을 단정하고 있다"


def test_uncalibrated_draws_no_number_even_if_level_text_is_set():
    """미보정인데 level_text 가 채워진 뷰가 와도 숫자를 그리지 않는다(이중 방어).

    viewmodel._level_text 하나에만 의존하면, 뷰를 다른 경로로 만든 순간 렌더러가
    " dB" 를 붙여 있지도 않은 물리 단위를 지어낸다.
    """
    lo = Layout.for_size(1280, 360)
    surf = _render(_view(level_text="97", spl_calibrated=False))
    band = [surf.get_at((x, y))[:3]
            for y in range(lo.radar_cy, min(360, lo.radar_cy + 34))
            for x in range(lo.db_right - 140, lo.db_right)]
    assert all(sum(c) < 60 for c in band), "미보정인데 숫자가 그려졌다"




# ---------------------------------------------------------------------------
# 레터박스 — 표면이 설계 비율(1280:360)보다 세로로 길면 띠를 가운데 둔다.
# 젯슨 실기에서 발견: 전체화면 표면이 요청보다 커도 좌표는 설정값으로 잡혀서
# 배경만 화면을 덮고 내용은 위쪽 360px 에 몰렸다.
# ---------------------------------------------------------------------------

def _render_at(w, h, view, frames=3):
    pygame.init()
    surf = pygame.Surface((w, h))
    r = Renderer(HudConfig(fullscreen=False))
    for _ in range(frames):
        r.draw(surf, view)
    return surf


def _content_rows(surf, thresh=150):
    w, h = surf.get_size()
    return [y for y in range(h)
            if any(sum(surf.get_at((x, y))[:3]) > thresh for x in range(0, w, 4))]


def test_taller_screen_centers_the_band_instead_of_stretching_it():
    """1280x800 에서 내용이 위로 몰리지도, 세로로 늘어나지도 않아야 한다."""
    rows = _content_rows(_render_at(1280, 800, _view()))
    assert rows, "아무것도 그려지지 않았다"
    band_top, band_bot = min(rows), max(rows)
    # 설계 띠(360)가 800 안에 들어가면 위아래 여백이 각각 220
    assert band_top >= 200, f"내용이 위로 몰렸다 (top={band_top})"
    assert band_bot <= 600, f"내용이 아래로 넘쳤다 (bottom={band_bot})"
    # 위아래 여백이 서로 비슷해야 '가운데'다
    assert abs(band_top - (800 - band_bot)) <= 60, \
        f"위아래 여백이 다르다 (위 {band_top}, 아래 {800 - band_bot})"


def test_reference_resolution_is_unchanged_by_the_letterbox_path():
    """1280x360 은 박스가 화면 전체라 기존 좌표 그대로여야 한다."""
    lo = Layout.for_size(1280, 360)
    surf = _render_at(1280, 360, _view())
    rows = _content_rows(surf)
    assert rows and min(rows) < lo.radar_cy < max(rows)


# ---------------------------------------------------------------------------
# 레이더 렌더 (v4) — 방향은 사분면 위치로, 음압은 켜진 링 양으로 확인한다
# ---------------------------------------------------------------------------

def _lit_by_quadrant(surf, lo, thresh=60):
    """레이더 영역을 상/하/좌/우로 나눠 **차종색** 픽셀 수를 센다.

    밝기로 세면 안 된다: 링 아래쪽 조명이 꺼진 회색 링까지 밝히므로 어떤 방향이든
    아래가 이긴다. 차종색은 채도가 있고(채널 최대-최소 > 60) 링은 회색이라 색으로
    가르면 조명과 무관하게 '켜진 아크'만 세어진다.
    """
    reach = lo.radar_ry + lo.ring_w + (5 - 1) * lo.ring_gap
    reach_x = lo.radar_rx + lo.ring_w + (5 - 1) * lo.ring_gap
    out = {"위": 0, "아래": 0, "좌": 0, "우": 0}
    for y in range(max(0, lo.radar_cy - reach), min(360, lo.radar_cy + reach)):
        for x in range(max(0, lo.radar_cx - reach_x), min(1280, lo.radar_cx + reach_x)):
            c = surf.get_at((x, y))[:3]
            if max(c) - min(c) <= thresh:      # 회색(링·배경) → 제외
                continue
            dy, dx = y - lo.radar_cy, x - lo.radar_cx
            if abs(dy) > abs(dx):
                out["위" if dy < 0 else "아래"] += 1
            else:
                out["좌" if dx < 0 else "우"] += 1
    return out


def test_each_direction_lights_its_own_side():
    lo = Layout.for_size(1280, 360)
    for direction, expect in [(Direction.FRONT, "위"), (Direction.REAR, "아래"),
                              (Direction.LEFT, "좌"), (Direction.RIGHT, "우")]:
        q = _lit_by_quadrant(_render(_view(direction=direction)), lo)
        top = max(q, key=q.get)
        assert top == expect, f"{direction} 인데 {top} 이 가장 밝다 ({q})"


def test_unknown_direction_spreads_evenly_and_differs_from_a_known_one():
    """모를 땐 네 방향이 고르게 — 한 방향만 켜서 '거기서 온다'고 말하면 안 된다."""
    lo = Layout.for_size(1280, 360)
    q = _lit_by_quadrant(_render(_view(direction=Direction.UNKNOWN)), lo)
    vals = sorted(q.values())
    assert vals[0] > 0, "미상인데 아무 아크도 안 켜졌다 — 미탐지와 구분이 안 된다"
    assert vals[-1] <= vals[0] * 2.2, f"미상인데 한쪽으로 쏠렸다 ({q})"
    right = _lit_by_quadrant(_render(_view(direction=Direction.RIGHT)), lo)
    assert right["우"] > q["우"], "미상과 우측이 같은 밝기다"


def test_louder_sound_lights_more_of_the_radar():
    lo = Layout.for_size(1280, 360)

    def lit(spl):
        surf = _render(_view(level_db=spl, level_text=f"{spl:.0f}"))
        return sum(_lit_by_quadrant(surf, lo).values())

    assert lit(105.0) > lit(88.0) > lit(66.0) > 0


def test_uncalibrated_radar_still_works_without_a_number():
    """보정 안 돼도 링은 돈다 — 소음계 없이도 이 디자인이 성립해야 한다."""
    lo = Layout.for_size(1280, 360)
    surf = _render(_view(level_db=-4.0, level_text=None, spl_calibrated=False))
    assert sum(_lit_by_quadrant(surf, lo).values()) > 0
    band = [surf.get_at((x, y))[:3]
            for y in range(max(0, lo.radar_cy - 30), min(360, lo.radar_cy + 30))
            for x in range(lo.db_right - 150, lo.db_right)]
    assert all(sum(c) < 90 for c in band), "미보정인데 숫자가 그려졌다"
