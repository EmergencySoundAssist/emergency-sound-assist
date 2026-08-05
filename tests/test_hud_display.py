"""HudDisplay 헤드리스 — SDL dummy로 _tick/stop 동작 확인."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pytest

pygame = pytest.importorskip("pygame")

from core.types import (
    FusedResult, ClassResult, DirectionResult, ApproachResult,
    SoundClass, Direction, Motion,
)
from hud.config import HudConfig
from hud.display import HudDisplay


def _fused():
    return FusedResult(
        sound=ClassResult.from_label(SoundClass.SIREN, 0.9),
        direction=DirectionResult(direction=Direction.REAR),
        approach=ApproachResult(motion=Motion.APPROACHING),
    )


def test_display_headless_idle_then_emergency():
    d = HudDisplay(HudConfig(width=320, height=180, fullscreen=False, fps=5))
    d._init_pygame()
    d._tick()                 # 대기(데이터 전)
    d.update(_fused())
    d._tick()                 # 긴급
    assert d.stopped is False
    d.stop()
    assert d.stopped is True
    pygame.quit()


def test_display_reflect_tick():
    d = HudDisplay(HudConfig(width=320, height=180, fullscreen=False,
                             reflect=True, fps=5))
    d._init_pygame()
    d.update(_fused())
    d._tick()                 # 반사(상하반전) 경로
    d.stop()
    pygame.quit()
