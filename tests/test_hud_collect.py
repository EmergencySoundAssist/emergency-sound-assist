"""수집 모드 HUD — 오버레이 렌더 스모크, 키 매핑, 버튼 터치 판정(반사 포함)."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pytest

pygame = pytest.importorskip("pygame")

from hud.config import HudConfig
from hud.display import _COLLECT_KEYS, HudDisplay
from hud.renderer import Renderer


class _FakeCollector:
    """HUD 가 쓰는 인터페이스(status/on_label/on_cancel)만 흉내낸다."""

    def __init__(self, recording=None, label=None):
        self._st = {"recording": recording, "elapsed": 7.0, "label": label,
                    "feedback": "라벨 ✓ 구급차 (직전 클립)", "counts": {}, "clips": 2,
                    "session": "20260811_1000_테스트"}
        self.labels = []
        self.cancels = 0

    def status(self):
        return dict(self._st)

    def on_label(self, subtype):
        self.labels.append(subtype)

    def on_cancel(self):
        self.cancels += 1


def test_overlay_draws_in_all_states_and_exposes_buttons():
    pygame.init()
    surf = pygame.Surface((1280, 360))
    r = Renderer(HudConfig(width=1280, height=360))
    r.draw(surf, None, collect=_FakeCollector().status())           # 대기 화면에도 버튼
    assert len(r.collect_buttons) == 6
    r.draw(surf, None, collect=_FakeCollector("auto", "fire").status())
    assert len(r.collect_buttons) == 6
    r.draw(surf, None)                                              # 수집 없으면 버튼 없음
    assert r.collect_buttons == []
    pygame.quit()


def test_key_mapping_matches_tag_siren_layout():
    assert _COLLECT_KEYS[pygame.K_1] == "ambulance"
    assert _COLLECT_KEYS[pygame.K_2] == "police"
    assert _COLLECT_KEYS[pygame.K_3] == "fire"
    assert _COLLECT_KEYS[pygame.K_u] == "unknown"
    assert _COLLECT_KEYS[pygame.K_KP1] == "ambulance"   # 키패드도 동작
    assert _COLLECT_KEYS[pygame.K_n] == "not_siren"     # 오검출 확인


def _display_with_fake(reflect=False):
    d = HudDisplay(HudConfig(width=320, height=180, fullscreen=False,
                             fps=5, reflect=reflect))
    fake = _FakeCollector()
    d.set_collector(fake)
    d._init_pygame()
    d._tick()               # 한 번 그려야 renderer 가 버튼 사각형을 기억한다
    return d, fake


def test_keydown_routes_to_collector():
    d, fake = _display_with_fake()
    for key in (pygame.K_1, pygame.K_u, pygame.K_n):
        d._collect_input(pygame.event.Event(pygame.KEYDOWN, key=key))
    d._collect_input(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_z))
    assert fake.labels == ["ambulance", "unknown", "not_siren"]
    assert fake.cancels == 1
    pygame.quit()


def test_click_on_button_labels_that_vehicle():
    d, fake = _display_with_fake()
    rect, key = d._renderer.collect_buttons[2]      # 세 번째 = fire
    assert key == "fire"
    d._collect_input(pygame.event.Event(
        pygame.MOUSEBUTTONDOWN, button=1, pos=rect.center))
    assert fake.labels == ["fire"]
    # 버튼 밖(좌상단 구석)은 무시
    d._collect_input(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(1, 1)))
    assert fake.labels == ["fire"]
    pygame.quit()


def test_synthetic_touch_mouse_event_is_ignored():
    """터치 1탭 = FINGERDOWN + 합성 MOUSEBUTTONDOWN(touch=True). 후자를 거르지
    않으면 한 탭이 라벨을 두 번 찍는다(직전 클립 라벨 + 유령 수동 녹음)."""
    d, fake = _display_with_fake()
    rect, _ = d._renderer.collect_buttons[0]
    d._collect_input(pygame.event.Event(
        pygame.MOUSEBUTTONDOWN, button=1, pos=rect.center, touch=True))
    assert fake.labels == []
    pygame.quit()


def test_click_hits_same_button_in_reflect_mode():
    """반사(상하반전) 모드: 화면에서 뒤집힌 위치를 눌러도 같은 버튼이어야 한다."""
    d, fake = _display_with_fake(reflect=True)
    rect, key = d._renderer.collect_buttons[0]
    h = d._screen.get_size()[1]
    flipped = (rect.centerx, h - 1 - rect.centery)  # 화면(뒤집힌)에서 보이는 위치
    d._collect_input(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=flipped))
    assert fake.labels == ["ambulance"]
    pygame.quit()
