"""HudView를 pygame Surface에 그린다 (고정 그리드: 좌측 텍스트 / 우측 바+음압 미터).

색/폰트는 여기 모아 둔다. 좌표·폰트 크기는 전부 card.Layout 에서 온다 — 상태가
바뀌어도 요소가 이동·신축하지 않게 하려면 배치 숫자가 한 곳에만 있어야 한다.
반사(상하반전)는 display 가 최종 Surface에 적용하므로 렌더러는 방향(위=전방)만
신경 쓴다.
"""

from __future__ import annotations

import os
import sys

import pygame

from hud import card as hud_card
from hud.card import Layout

# HUD 팔레트 (차량용 다크 화면 — 모드 적응 불필요, 고정)
BG = (10, 10, 11)
FG = (245, 245, 245)
MUTED = (90, 90, 94)
ALERT = (226, 75, 74)      # 긴급(빨강)
WARN = (239, 159, 39)      # 접근·보조(주황)
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
# 세그먼트 높이·글로우 배수는 이제 Layout.seg_h / card.spl_to_glow 가 정한다.
SPREAD = 2              # 정적 점등 반경 — 방향만 나타낸다(음압과 무관)
RIPPLE_RADIUS = 4       # 퍼짐이 도달하는 최대 반경. 밝기만 번지고 칸은 불변


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
        self._f_mid = load_font(max(18, h // 12), config.font_path)   # 대기 문구
        self._f_sub = load_font(max(20, h // 9), config.font_path)    # 점 애니메이션 크기
        self._frame = 0
        # 고정 그리드 폰트 — 크기도 Layout 에서 나온다. 좌표만 모으고 폰트를 여기
        # 남기면 둘이 따로 놀아 그리드가 어긋난다(차종 104px 자리에 40px 글자).
        self._lo = Layout.for_size(config.width, config.height)
        lo = self._lo
        self._f_veh = load_font(lo.f_veh, config.font_path)      # 차종 — 압도적으로 크게
        self._f_state = load_font(lo.f_state, config.font_path)  # 접근/멀어짐
        self._f_db = load_font(lo.f_db, config.font_path)        # 음압 숫자(보조)
        self._f_unit = load_font(lo.f_unit, config.font_path)    # dB 단위 · L/R 라벨
        self._f_cap = load_font(lo.f_cap, config.font_path)      # 자막

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
        """고정 그리드: 왼쪽 텍스트 / 오른쪽 바+미터. 좌표는 Layout 에서만 온다."""
        self._frame = (self._frame + 1) % 100000
        lo = self._lo
        color = vehicle_color(view.sound_text)
        t = hud_card.spl_intensity(view.level_db, view.spl_calibrated)

        surface.blit(self._f_veh.render(view.sound_text, True, FG), lo.veh_xy)
        front = not self._bar_visible(view)
        state = f"{view.direction_text} · {view.motion_text}" if front else view.motion_text
        if getattr(view, "is_horn", False):
            state = view.direction_text
        surface.blit(self._f_state.render(state, True, color), lo.state_xy)

        if front:
            self._draw_bar(surface, color, 0, t, lit=False)
            mark = self._f_unit.render("▲ 전방", True, color)
            x0, total, _, _ = self._bar_span()
            # 칸 '위'에 띄운다 — 칸을 덮으면 고정 그리드가 깨져 보인다.
            # 간격도 Layout(칸 높이)에서 나온다: 화면이 커지면 같이 벌어진다.
            surface.blit(mark, mark.get_rect(
                midbottom=(x0 + total // 2, lo.bar_cy - lo.seg_h)))
        else:
            self._draw_bar(surface, color, self._bar_center(view), t,
                           ripple=hud_card.should_ripple(
                               getattr(view, "is_horn", False), view.approach_motion()))

        self._draw_meter(surface, color, t)
        if view.level_text:
            self._draw_db(surface, view.level_text, color)
        if view.subtitle:
            lines = hud_card.wrap_text(
                view.subtitle, lambda s: self._f_cap.size(s)[0],
                w - 2 * lo.margin, max_lines=2)
            self._center_multiline(surface, lines, self._f_cap, FG, w // 2, lo.cap_cy)

    def _bar_visible(self, view) -> bool:
        if view.angle_deg is not None:
            return hud_card.azimuth_to_bar_index(view.angle_deg, self._lo.seg_n) is not None
        return hud_card.direction_visible(view.direction)

    def _bar_center(self, view) -> int:
        if view.angle_deg is not None:
            idx = hud_card.azimuth_to_bar_index(view.angle_deg, self._lo.seg_n)
            if idx is not None:
                return idx
        return hud_card.direction_to_index(None, view.direction, self._lo.seg_n)

    def _bar_span(self):
        """칸 피치(칸 폭·간격)의 유일한 계산처 — x0, 전체 폭, 칸 폭, 간격.

        여기서만 계산해야 한다: 두 곳에서 따로 계산하면 한쪽만 바뀔 때 칸과
        L/R 라벨이 어긋난다.
        """
        lo = self._lo
        seg_w = int(lo.bar_w / (lo.seg_n + (lo.seg_n - 1) * 0.28))
        gap = int(seg_w * 0.28)
        total = lo.seg_n * seg_w + (lo.seg_n - 1) * gap
        return lo.bar_x + (lo.bar_w - total) // 2, total, seg_w, gap

    def _draw_bar(self, surface, color, center, t, lit=True, ripple=False):
        """고정 그리드 LED. 칸의 위치·크기는 불변, 밝기만 바뀐다."""
        lo = self._lo
        x0, total, seg_w, gap = self._bar_span()
        y = lo.bar_cy - lo.seg_h // 2
        glow = hud_card.spl_to_glow(t)
        period = hud_card.ripple_period_for_spl(t)
        for i in range(lo.seg_n):
            if not lit:
                b = 0.0
            elif ripple:
                b = hud_card.ripple_brightness(i, center, RIPPLE_RADIUS,
                                               self._frame, period)
            else:
                b = hud_card.segment_brightness(i, center, SPREAD)
            _glow_segment(surface, x0 + i * (seg_w + gap), y, seg_w, lo.seg_h,
                          color, b, glow)
        label_gap = lo.margin // 3   # margin 의 1/3 — 1280px 기준 기존 24px 그대로
        ll = self._f_unit.render("L", True, MUTED)
        surface.blit(ll, (x0 - ll.get_width() - label_gap, lo.bar_cy - ll.get_height() // 2))
        rl = self._f_unit.render("R", True, MUTED)
        surface.blit(rl, (x0 + total + label_gap, lo.bar_cy - rl.get_height() // 2))

    def _draw_meter(self, surface, color, t):
        """음압 고정 트랙 미터. 트랙 길이는 불변, 채워지는 길이만 변한다."""
        lo = self._lo
        x0, total, _, _ = self._bar_span()
        pygame.draw.rect(surface, DIM, (x0, lo.meter_y, total, lo.meter_h),
                         border_radius=6)
        if t > 0:
            pygame.draw.rect(surface, color,
                             (x0, lo.meter_y, max(6, int(total * t)), lo.meter_h),
                             border_radius=6)

    def _draw_db(self, surface, text, color):
        lo = self._lo
        num = self._f_db.render(text, True, color)
        unit = self._f_unit.render(" dB", True, MUTED)
        x = lo.db_right - num.get_width() - unit.get_width()
        surface.blit(num, (x, lo.db_y))
        surface.blit(unit, (x + num.get_width(),
                            lo.db_y + num.get_height() - unit.get_height() - 3))

    def _draw_normal(self, surface, view, w, h):
        """평상시도 같은 그리드 — 긴급 전환 시 화면이 재배치되지 않는다."""
        self._frame = (self._frame + 1) % 100000
        lo = self._lo
        surface.blit(self._f_veh.render("듣는 중", True, MUTED), lo.veh_xy)
        self._draw_bar(surface, MUTED, 0, 0.0, lit=False)
        self._draw_meter(surface, MUTED, 0.0)     # 빈 트랙 — 긴급과 같은 자리·같은 길이
        if view.subtitle:
            lines = hud_card.wrap_text(
                view.subtitle, lambda s: self._f_cap.size(s)[0],
                w - 2 * lo.margin, max_lines=2)
            self._center_multiline(surface, lines, self._f_cap, FG, w // 2, lo.cap_cy)
        else:
            self._draw_dots(surface, w // 2, lo.cap_cy)

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
