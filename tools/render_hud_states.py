"""HUD 상태별 오프스크린 렌더 → PNG. 디자인 검토·회귀 눈확인용.

사용: python tools/render_hud_states.py [출력디렉터리]
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

# repo 루트를 sys.path 에 올린다 — `python tools/render_hud_states.py` 로 실행하면
# 파이썬이 tools/ 만 sys.path[0]에 넣어서 core/hud 임포트가 실패한다.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame

from core.types import Direction, Motion
from hud.config import HudConfig
from hud.renderer import Renderer
from hud.viewmodel import HudView


def _view(**kw):
    base = dict(
        emergency=True, sound_text="구급차", direction=Direction.RIGHT,
        direction_text="우측", motion_text="접근 중", subtitle="",
        confidence=0.9, angle_deg=None, motion=Motion.APPROACHING,
        is_horn=False, level_db=97.0, level_text="97", spl_calibrated=True,
    )
    base.update(kw)
    return HudView(**base)


STATES = {
    "01_우측_97db": _view(),
    "02_후방_79db": _view(direction=Direction.REAR, direction_text="후방",
                          level_db=79.0, level_text="79"),
    "03_좌측_66db_멀어짐": _view(direction=Direction.LEFT, direction_text="좌측",
                                 motion=Motion.RECEDING, motion_text="멀어짐",
                                 level_db=66.0, level_text="66"),
    "04_전방": _view(direction=Direction.FRONT, direction_text="전방"),
    "05_경적": _view(sound_text="경적", is_horn=True, motion=Motion.UNKNOWN,
                     motion_text="이동 미상", level_db=88.0, level_text="88"),
    # ★ 미보정 — 스펙 §11 이 남긴 눈금 확인 지점. dBFS 실측 분포에서 미터가
    #   바닥이나 끝에 붙어 있으면 card.DBFS_RANGE 를 다시 잡아야 한다.
    "06_미보정_-4dBFS": _view(level_db=-4.0, level_text=None, spl_calibrated=False),
    "07_미보정_-30dBFS": _view(level_db=-30.0, level_text=None, spl_calibrated=False),
    "08_자막동반": _view(subtitle="구급차가 지나갑니다"),
    "09_평상시_자막": _view(emergency=False, sound_text="일반 도로 소음",
                            motion=Motion.UNKNOWN, motion_text="이동 미상",
                            subtitle="잠시만요, 지금 길 좀 비켜 주시겠어요?"),
    "10_평상시_대기": _view(emergency=False, sound_text="일반 도로 소음",
                            motion=Motion.UNKNOWN, motion_text="이동 미상"),
}


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out, exist_ok=True)
    pygame.init()
    cfg = HudConfig(fullscreen=False)
    surf = pygame.Surface((cfg.width, cfg.height))
    r = Renderer(cfg)
    for name, view in STATES.items():
        for _ in range(4):            # 퍼짐 위상을 조금 진행시킨 프레임
            r.draw(surf, view)
        pygame.image.save(surf, os.path.join(out, f"hud_{name}.png"))
        print("saved", name)


if __name__ == "__main__":
    main()
