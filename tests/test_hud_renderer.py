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


def test_meter_track_does_not_move_between_states():
    """긴급 ↔ 평상시에 음압 트랙이 같은 자리·같은 길이 — 눈이 매번 같은 곳을 본다.

    바(칸)는 글로우 헤일로가 상태마다 번지는 폭이 달라 픽셀로 비교할 수 없다.
    트랙은 글로우가 없어 '움직이지 않음'을 그대로 확인할 수 있는 자리다.
    """
    lo = Layout.for_size(1280, 360)
    row = lo.meter_y + lo.meter_h // 2

    def occupied(view):
        surf = _render(view)
        return [x for x in range(lo.bar_x, 1280)
                if surf.get_at((x, row))[:3] != (10, 10, 11)]

    assert occupied(_view()) == occupied(_view(emergency=False)) != []


def test_meter_fills_more_for_a_louder_sound():
    lo = Layout.for_size(1280, 360)
    row = lo.meter_y + lo.meter_h // 2

    def filled(spl):
        surf = _render(_view(level_db=spl, level_text=f"{spl:.0f}"))
        return sum(1 for x in range(1280) if sum(surf.get_at((x, row))[:3]) > 90)

    assert filled(97.0) > filled(66.0) > 0


def test_uncalibrated_view_draws_no_number():
    """미보정이면 숫자를 아예 그리지 않는다 — 단위를 속이지 않는다."""
    lo = Layout.for_size(1280, 360)
    surf = _render(_view(level_text=None, spl_calibrated=False))
    band = [surf.get_at((x, y))[:3]
            for y in range(lo.db_y, min(360, lo.db_y + 34))
            for x in range(lo.db_right - 140, lo.db_right)]
    assert all(sum(c) < 60 for c in band), "미보정인데 숫자가 그려졌다"


def test_front_lights_no_segment_but_still_marks_itself():
    """전방은 칸을 켜지 않는다. 그래도 '감지 없음'과 구분돼야 한다."""
    lo = Layout.for_size(1280, 360)
    row = lo.bar_cy
    front = _render(_view(direction=Direction.FRONT, direction_text="전방"))
    lit = sum(1 for x in range(lo.bar_x, lo.bar_x + lo.bar_w)
              if sum(front.get_at((x, row))[:3]) > 150)
    assert lit == 0, "전방인데 칸이 켜졌다"
    above = sum(1 for x in range(lo.bar_x, lo.bar_x + lo.bar_w)
                for y in range(lo.bar_cy - lo.seg_h // 2 - 34, lo.bar_cy - lo.seg_h // 2)
                if sum(front.get_at((x, y))[:3]) > 150)
    assert above > 0, "전방 표식이 없다"
