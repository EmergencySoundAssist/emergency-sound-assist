"""Renderer 헤드리스 렌더 스모크 — SDL dummy, 예외 없이 그려지면 통과."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pytest

pygame = pytest.importorskip("pygame")

from core.types import Direction, Motion
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
# 서브라인 — 거리 문구와 거리비
# ---------------------------------------------------------------------------
def _view(**kw):
    base = dict(emergency=True, sound_text="구급차", direction=Direction.REAR,
                direction_text="후방", motion_text="멀어짐", subtitle="",
                confidence=0.9, motion=Motion.RECEDING)
    base.update(kw)
    return HudView(**base)


def test_subline_appends_relative_distance_with_an_explicit_reference():
    """미터로 오해되지 않게 '최근접 대비'를 반드시 붙인다."""
    line = Renderer._subline(_view(proximity="원거리", rel_distance=3.32))
    assert "최근접 대비 3.3배" in line
    assert "원거리" in line


def test_subline_caps_a_runaway_ratio():
    line = Renderer._subline(_view(proximity="원거리", rel_distance=27.4))
    assert "10배+" in line and "27" not in line


def test_subline_omits_distance_while_still_approaching():
    """접근 중에는 거리비가 없다 — 없는 값을 지어내지 않는다."""
    line = Renderer._subline(_view(motion=Motion.APPROACHING, motion_text="접근 중",
                                   proximity="근접", rel_distance=None))
    assert "배" not in line
    assert line.endswith("근접")
