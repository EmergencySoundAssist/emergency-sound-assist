"""HUD 설정 — pygame 의존 없음(테스트·main 양쪽에서 자유롭게 import)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class HudConfig:
    """HUD 렌더 설정.

    fullscreen: 전체화면(기본). 노트북 개발은 False(창).
    reflect   : 반사(윈드실드) 모드 = 상하반전. 런타임 F키로 토글.
    fps       : 렌더 프레임레이트(청크 1초 주기와 무관하게 매끄럽게).
    font_path : 한국어 폰트 경로 강제. None이면 후보 경로 자동 탐색.
    """
    width: int = 1280
    height: int = 720
    fullscreen: bool = True
    reflect: bool = False
    fps: int = 30
    font_path: Optional[str] = None
