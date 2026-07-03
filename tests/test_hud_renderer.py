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
