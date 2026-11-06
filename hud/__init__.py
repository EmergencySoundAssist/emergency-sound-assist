"""HUD 출력 모듈 — FusedResult를 화면(pygame)에 그린다.

pygame 의존 심볼(HudDisplay)은 여기서 export하지 않는다. main.py가 --hud일 때만
`from hud.display import HudDisplay`로 지연 import해 pygame 미설치 환경을 깨지 않는다.
"""

from hud.config import HudConfig
from hud.viewmodel import HudView

__all__ = ["HudConfig", "HudView"]
