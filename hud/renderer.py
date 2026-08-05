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
WARN = (239, 159, 39)      # 접근·보조(주황)
DIM = (35, 35, 38)

# 차종별 색 (LED 카드)
VEH_AMBULANCE = (51, 220, 90)     # 구급차 초록
VEH_POLICE = (60, 130, 246)       # 경찰차 파랑
VEH_FIRE = (226, 75, 74)          # 소방차 빨강(긴급색)
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
        self._font_path = config.font_path
        self._frame = 0
        self._size = None                       # 마지막으로 레이아웃을 잡은 박스 크기
        # 설계 비율(기본 1280:360 ≈ 3.56:1). 윈드실드 띠를 전제로 잡은 값이라,
        # 화면이 더 정사각형에 가까워도 이 비율을 유지해야 배치가 무너지지 않는다.
        self._aspect = config.width / max(1, config.height)
        self._build(config.width, config.height)

    def _design_box(self, w: int, h: int) -> "pygame.Rect":
        """설계 비율을 지킨 채 화면 안에 최대로 내접시키고 가운데 놓는다.

        전체화면에서 실제 표면이 16:10 처럼 세로로 길면, 비율을 무시하고 좌표를 늘리면
        상태 문구가 미터와 겹치고 간격만 벌어진다(젯슨 실기에서 확인). 띠를 그대로 두고
        위아래를 배경으로 남기는 편이 읽기에 낫다.
        """
        if w / max(1, h) > self._aspect:        # 화면이 설계보다 넓다 → 높이에 맞춤
            bw, bh = int(h * self._aspect), h
        else:                                   # 화면이 설계보다 높다 → 폭에 맞춤
            bw, bh = w, int(w / self._aspect)
        return pygame.Rect((w - bw) // 2, (h - bh) // 2, bw, bh)

    def _build(self, w: int, h: int) -> None:
        """(w, h) 에 맞춰 좌표와 폰트를 다시 잡는다.

        설정값이 아니라 **실제로 그릴 표면 크기**를 기준으로 해야 한다. 전체화면에서
        SDL 이 요청한 1280×360 모드를 못 주면 더 큰 표면을 돌려주는데, 그때 설정값으로
        좌표를 잡으면 배경만 화면을 덮고 내용은 위쪽 360px 에 몰린다(젯슨 실기에서 확인).
        """
        self._size = (w, h)
        self._lo = Layout.for_size(w, h)
        lo = self._lo
        self._f_mid = load_font(max(18, h // 12), self._font_path)   # 대기 문구
        self._f_sub = load_font(max(20, h // 9), self._font_path)    # 점 애니메이션 크기
        # 고정 그리드 폰트 — 크기도 Layout 에서 나온다. 좌표만 모으고 폰트를 여기
        # 남기면 둘이 따로 놀아 그리드가 어긋난다(차종 104px 자리에 40px 글자).
        self._f_veh = load_font(lo.f_veh, self._font_path)      # 차종 — 압도적으로 크게
        self._f_state = load_font(lo.f_state, self._font_path)  # 접근/멀어짐
        self._f_db = load_font(lo.f_db, self._font_path)        # 음압 숫자(보조)
        self._f_unit = load_font(lo.f_unit, self._font_path)    # dB 단위 · L/R 라벨
        self._f_cap = load_font(lo.f_cap, self._font_path)      # 자막

    def draw(self, surface, view) -> None:
        surface.fill(BG)                # 레터박스 여백도 같은 배경으로 덮는다
        box = self._design_box(*surface.get_size())
        if box.size != self._size:      # 크기가 바뀐 프레임에만 재계산(폰트 로드가 비싸다)
            self._build(*box.size)
        # 박스 안에서만 그린다. subsurface 는 픽셀을 공유하고 좌표가 박스 기준이라
        # 아래 그리기 코드는 화면이 얼마나 크든 1280×360 을 그린다고 믿으면 된다.
        target = surface.subsurface(box) if box.size != surface.get_size() else surface
        w, h = box.size
        if view is None:
            self._center(target, "연결됨 · 대기 중", self._f_mid, MUTED, w // 2, h // 2)
            return
        if view.emergency:
            self._draw_emergency(target, view, w, h)
        else:
            self._draw_normal(target, view, w, h)

    def _draw_emergency(self, surface, view, w, h):
        """고정 그리드: 왼쪽 텍스트 / 오른쪽 바+미터. 좌표는 Layout 에서만 온다."""
        self._frame = (self._frame + 1) % 100000
        lo = self._lo
        color = vehicle_color(view.sound_text)
        t = hud_card.spl_intensity(view.level_db, view.spl_calibrated)

        surface.blit(self._f_veh.render(view.sound_text, True, FG), lo.veh_xy)
        front = not self._bar_visible(view)
        # 방향은 항상 문구로 낸다. 칸만으로는 후방과 방향 미상을 구분할 수 없다 —
        # direction_to_index 가 둘 다 중앙(7)이라 픽셀이 완전히 같아진다. 문구가
        # 없으면 방향을 모를 때도 "후방에서 온다"고 단정하는 화면이 되는데, 소리를
        # 못 듣는 운전자에게 이 화면이 유일한 경고다. 모르면 모른다고 써야 한다.
        state = view.direction_text if getattr(view, "is_horn", False) \
            else f"{view.direction_text} · {view.motion_text}"
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
        # spl_calibrated 를 여기서도 본다(viewmodel._level_text 와 이중 방어).
        # _draw_db 는 " dB" 를 무조건 붙이므로, 한 곳만 지키면 뷰가 잘못 만들어졌을 때
        # 없는 단위를 지어내 출력한다 — '단위를 속이지 않는다'는 제약의 핵심이다.
        if view.level_text and view.spl_calibrated:
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
