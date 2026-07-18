"""HudView를 pygame Surface에 그린다 (A안: 방향 레이더 + 평상시 자막).

색/폰트는 여기 모아 둔다. 반사(상하반전)는 display 가 최종 Surface에 적용하므로
렌더러는 방향(위=전방)만 신경 쓴다.
"""

from __future__ import annotations

import os
import sys

import pygame

from core.types import Direction
from hud import card as hud_card

# HUD 팔레트 (차량용 다크 화면 — 모드 적응 불필요, 고정)
BG = (10, 10, 11)
FG = (245, 245, 245)
MUTED = (90, 90, 94)
ALERT = (226, 75, 74)      # 긴급(빨강)
WARN = (239, 159, 39)      # 접근·보조(주황)
PANEL = (20, 20, 22)
DIM = (35, 35, 38)

# 차종별 색 (LED 카드)
VEH_AMBULANCE = (51, 220, 90)     # 구급차 초록
VEH_POLICE = (60, 130, 246)       # 경찰차 파랑
VEH_FIRE = (226, 75, 74)          # 소방차 빨강(=ALERT)
VEH_OTHER = (239, 159, 39)        # 기타/경적 주황(=WARN)


def vehicle_color(sound_text: str) -> tuple:
    """차종 텍스트 → LED 색. 미상/경적 등은 기타색."""
    if "구급" in sound_text:
        return VEH_AMBULANCE
    if "경찰" in sound_text:
        return VEH_POLICE
    if "소방" in sound_text:
        return VEH_FIRE
    return VEH_OTHER


# LED 스트립 디자인 확정값 (디자인 스튜디오에서 튜닝)
GLOW = 2.0                    # 세그먼트 번짐 강도 배수 (1.0 기본, 2.0 = 네온)
SEG_H_RATIO = 26.0 / 360.0    # 세그먼트 높이 비율 (≈26px @360)


def _glow_segment(surface, x, y, w, h, color, b, glow=1.0):
    """밝기 b(0~1)로 LED 세그먼트 + 글로우. b<=0이면 꺼진 세그먼트(DIM).

    glow: 번짐 강도 배수 — 헤일로의 크기·밝기를 함께 키운다.
    """
    if b <= 0.0:
        pygame.draw.rect(surface, DIM, (x, y, w, h), border_radius=5)
        return
    if glow > 0:
        pad = int(12 * glow)
        alpha = min(255, int(70 * b * glow))
        g = pygame.Surface((w + 2 * pad, h + 2 * pad), pygame.SRCALPHA)
        pygame.draw.rect(g, (*color, alpha),
                         (0, 0, w + 2 * pad, h + 2 * pad), border_radius=12)
        surface.blit(g, (x - pad, y - pad))
    col = tuple(int(c * (0.4 + 0.6 * b)) for c in color)
    pygame.draw.rect(surface, col, (x, y, w, h), border_radius=5)

# repo 번들 폰트(Pretendard SemiBold, OFL) — Mac·Jetson 동일 렌더 보장, 최우선.
_BUNDLED_FONT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets", "fonts", "Pretendard-SemiBold.otf",
)

# 한국어 폰트 후보 (없으면 다음 후보 → 최후에 기본 폰트+경고)
_FONT_CANDIDATES = [
    _BUNDLED_FONT,
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
        # 자동차 대시보드 카드 전용 폰트 계층
        self._f_veh = load_font(max(40, h // 12), config.font_path)   # 차종명(큰 글씨)
        self._f_line = load_font(max(20, h // 26), config.font_path)  # 방향·상태 서브라인
        self._f_tag = load_font(max(14, h // 40), config.font_path)   # 소label/LR
        self._f_cap = load_font(max(30, h // 14), config.font_path)   # 하단 자막
        self._frame = 0

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
        """자동차 대시보드 카드: 헤더 계층 / 글로우 LED 스트립 / 하단 구분선+자막."""
        self._frame = (self._frame + 1) % 100000
        color = vehicle_color(view.sound_text)
        m = int(w * 0.0625)                      # 좌우 여백(1280 기준 80)

        # 헤더: 작은 라벨 + 큰 차종명 + 강조선 + 방향·상태 서브라인
        self._text(surface, "EMERGENCY VEHICLE", self._f_tag, MUTED, m, int(h * 0.097))
        self._text(surface, view.sound_text, self._f_veh, FG, m, int(h * 0.133))
        line_y = int(h * 0.133) + self._f_veh.get_height() + 6
        pygame.draw.rect(surface, color, (m, line_y, int(w * 0.094), 3))
        self._text(surface, self._subline(view), self._f_line, color, m, line_y + 10)

        # 중앙: 방향 LED 스트립 — 각도(azimuth) 실시간 구동. 전방이면 바 자체를 숨긴다.
        if view.angle_deg is not None:
            idx = hud_card.azimuth_to_bar_index(view.angle_deg, n=15)
            show_dir, center = (idx is not None), (idx if idx is not None else 7)
        else:                                      # 각도 없음 → 이산 방향 폴백
            show_dir = hud_card.direction_visible(view.direction)
            center = hud_card.direction_to_index(None, view.direction, n=15)
        if show_dir:
            radius = hud_card.spread_for_speed(view.speed_level)   # 접근 빠를수록 넓게
            ripple = hud_card.should_ripple(getattr(view, "is_horn", False),
                                            view.approach_motion())
            self._draw_direction_strip(surface, w, h, h // 2, color, center,
                                       radius, ripple, self._frame, GLOW)

        # 하단: 구분선 + 자막(있을 때만, 자동 줄바꿈)
        if view.subtitle:
            pygame.draw.rect(surface, DIM, (m, h - int(h * 0.18), w - 2 * m, 2))
            lines = hud_card.wrap_text(
                view.subtitle, lambda s: self._f_cap.size(s)[0],
                w - 2 * m, max_lines=2,
            )
            self._center_multiline(surface, lines, self._f_cap, FG, w // 2, h - int(h * 0.1))

    def _draw_direction_strip(self, surface, w, h, y, color, center,
                              radius, ripple, frame, glow):
        """15-세그먼트 방향 LED 스트립 + L/R 라벨을 행 y 에 그린다.

        center: 점등 중심 세그먼트 인덱스. radius: 퍼짐 반경(접근 빠르기가 결정).
        ripple=True 면 중앙→바깥으로 퍼지는 깜빡임, False 면 정적 클러스터.
        긴급/평상시(idle) 화면이 공유한다.
        """
        n = 15
        seg_w = int(w * 0.042)
        gap = int(seg_w * 0.26)
        total = n * seg_w + (n - 1) * gap
        x0 = (w - total) // 2
        seg_h = int(h * SEG_H_RATIO)
        for i in range(n):
            if ripple:
                b = hud_card.ripple_brightness(i, center, radius, frame)
            else:
                b = hud_card.segment_brightness(i, center, radius)
            _glow_segment(surface, x0 + i * (seg_w + gap), y, seg_w, seg_h, color, b, glow)
        ll = self._f_tag.render("L", True, MUTED)
        surface.blit(ll, (x0 - ll.get_width() - 18, y + seg_h // 2 - ll.get_height() // 2))
        rl = self._f_tag.render("R", True, MUTED)
        surface.blit(rl, (x0 + total + 18, y + seg_h // 2 - rl.get_height() // 2))

    @staticmethod
    def _subline(view) -> str:
        """헤더 서브라인. 경적은 접근/이동을 출력하지 않고 방향만 보여준다."""
        known = (Direction.LEFT, Direction.RIGHT, Direction.FRONT, Direction.REAR)
        if getattr(view, "is_horn", False):
            return view.direction_text if view.direction in known else ""
        if view.direction in known:
            return f"{view.direction_text}에서 {view.motion_text}"
        return view.motion_text

    def _draw_normal(self, surface, view, w, h):
        self._frame = (self._frame + 1) % 100000
        self._text(surface, view.sound_text, self._f_mid, MUTED, 24, 24)

        # 방향 바(idle) — STT 모드에서도 항상 표시(중립색, 중앙, 정적, 은은한 글로우)
        center = hud_card.direction_to_index(None, Direction.UNKNOWN, n=15)
        self._draw_direction_strip(surface, w, h, int(h * 0.42), MUTED, center,
                                   3, False, self._frame, 1.0)

        # 하단 밴드: 자막(자동 줄바꿈) 또는 변환 중 점 애니메이션
        band_h = h // 3
        pygame.draw.rect(surface, PANEL, pygame.Rect(0, h - band_h, w, band_h))
        cy = h - band_h // 2
        if view.subtitle:
            lines = hud_card.wrap_text(
                view.subtitle, lambda s: self._f_sub.size(s)[0],
                int(w * 0.9), max_lines=2,
            )
            self._center_multiline(surface, lines, self._f_sub, FG, w // 2, cy)
        else:
            self._draw_dots(surface, w // 2, cy)

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

    def _center_multiline(self, surface, lines, font, color, cx, cy):
        """여러 줄을 (cx, cy) 세로 중앙 기준으로 쌓아 그린다."""
        if not lines:
            return
        lh = font.get_height()
        y = cy - (lh * len(lines)) // 2
        for ln in lines:
            surf = font.render(ln, True, color)
            surface.blit(surf, surf.get_rect(center=(cx, y + lh // 2)))
            y += lh

    def _draw_dots(self, surface, cx, cy):
        """STT 변환 중 로딩 점 3개 — 밝기 웨이브 + 살짝 위로 튀는 움직임."""
        bs = hud_card.dots_brightness(self._frame)
        r, gap = max(5, self._f_sub.get_height() // 6), 34
        x0 = cx - gap
        for i, b in enumerate(bs):
            col = tuple(int(c * (0.3 + 0.7 * b)) for c in FG)
            y = cy - int(b * 8)                       # 밝을수록 위로(움직임)
            pygame.draw.circle(surface, col, (x0 + i * gap, y), r)
