"""HudView를 pygame Surface에 그린다 (A안: 방향 레이더 + 평상시 자막).

색/폰트는 여기 모아 둔다. 반사(상하반전)는 display 가 최종 Surface에 적용하므로
렌더러는 방향(위=전방)만 신경 쓴다.
"""

from __future__ import annotations

import os
import sys

import pygame

from core.types import Direction

# HUD 팔레트 (차량용 다크 화면 — 모드 적응 불필요, 고정)
BG = (10, 10, 11)
FG = (245, 245, 245)
MUTED = (90, 90, 94)
ALERT = (226, 75, 74)      # 긴급(빨강)
WARN = (239, 159, 39)      # 접근·보조(주황)
PANEL = (20, 20, 22)
DIM = (35, 35, 38)

# 한국어 폰트 후보 (없으면 다음 후보 → 최후에 기본 폰트+경고)
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
]


def load_font(size: int, path=None) -> "pygame.font.Font":
    """한국어 렌더 가능한 폰트 로드. 못 찾으면 기본 폰트(경고)."""
    for p in ([path] if path else []) + _FONT_CANDIDATES:
        if p and os.path.exists(p):
            try:
                return pygame.font.Font(p, size)
            except Exception:
                continue
    print("[hud] 경고: 한국어 폰트 없음 → 기본 폰트(한글 □ 가능). "
          "sudo apt install fonts-nanum", file=sys.stderr)
    return pygame.font.Font(None, size)


class Renderer:
    def __init__(self, config):
        h = config.height
        self._f_sound = load_font(max(24, h // 6), config.font_path)
        self._f_mid = load_font(max(18, h // 12), config.font_path)
        self._f_small = load_font(max(12, h // 24), config.font_path)
        self._f_sub = load_font(max(20, h // 9), config.font_path)

    def draw(self, surface, view) -> None:
        w, h = surface.get_size()
        surface.fill(BG)
        if view is None:
            self._center(surface, "연결됨 · 대기 중", self._f_mid, MUTED, w // 2, h // 2)
            return
        if view.emergency:
            self._draw_emergency(surface, view, w, h)
        else:
            self._draw_normal(surface, view, w, h)

    def _draw_emergency(self, surface, view, w, h):
        side = min(h - 40, w // 2 - 40)
        radar = pygame.Rect(20, (h - side) // 2, side, side)
        self._draw_radar(surface, radar, view.direction)
        tx = radar.right + 30
        cy = h // 2
        self._text(surface, view.sound_text, self._f_sound, FG, tx,
                   cy - self._f_sound.get_height())
        self._text(surface, f"{view.direction_text} · {view.motion_text}",
                   self._f_mid, WARN, tx, cy + 10)
        self._text(surface, "자막 — 긴급 중 일시정지", self._f_small, MUTED, tx, h - 40)

    def _draw_normal(self, surface, view, w, h):
        self._text(surface, view.sound_text, self._f_mid, MUTED, 24, 24)
        band_h = h // 3
        pygame.draw.rect(surface, PANEL, pygame.Rect(0, h - band_h, w, band_h))
        text = view.subtitle if view.subtitle else "…"
        color = FG if view.subtitle else MUTED
        self._center(surface, text, self._f_sub, color, w // 2, h - band_h // 2)

    def _draw_radar(self, surface, rect, active):
        pygame.draw.rect(surface, DIM, rect, width=2, border_radius=12)
        sectors = {
            Direction.FRONT: pygame.Rect(rect.centerx - 30, rect.top + 12, 60, 34),
            Direction.REAR: pygame.Rect(rect.centerx - 30, rect.bottom - 46, 60, 34),
            Direction.LEFT: pygame.Rect(rect.left + 12, rect.centery - 17, 34, 34),
            Direction.RIGHT: pygame.Rect(rect.right - 46, rect.centery - 17, 34, 34),
        }
        labels = {Direction.FRONT: "전", Direction.REAR: "후",
                  Direction.LEFT: "좌", Direction.RIGHT: "우"}
        for d, r in sectors.items():
            on = (d == active)
            pygame.draw.rect(surface, ALERT if on else DIM, r, border_radius=8)
            self._center(surface, labels[d], self._f_small, BG if on else MUTED,
                         r.centerx, r.centery)
        car = pygame.Rect(0, 0, 34, 52)
        car.center = rect.center
        pygame.draw.rect(surface, DIM, car, border_radius=8)

    def _text(self, surface, s, font, color, x, y):
        surface.blit(font.render(s, True, color), (x, y))

    def _center(self, surface, s, font, color, cx, cy):
        surf = font.render(s, True, color)
        surface.blit(surf, surf.get_rect(center=(cx, cy)))
