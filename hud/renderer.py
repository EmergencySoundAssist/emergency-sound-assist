"""HudView를 pygame Surface에 그린다 (고정 그리드: 좌측 텍스트 / 우측 바+음압 미터).

색/폰트는 여기 모아 둔다. 좌표·폰트 크기는 전부 card.Layout 에서 온다 — 상태가
바뀌어도 요소가 이동·신축하지 않게 하려면 배치 숫자가 한 곳에만 있어야 한다.
반사(상하반전)는 display 가 최종 Surface에 적용하므로 렌더러는 방향(위=전방)만
신경 쓴다.
"""

from __future__ import annotations

import math
import os
import sys

import pygame

from core.types import Direction
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
# 레이더 중앙 차 — 위에서 본 형태. 지붕이 밝고 유리가 어두워야 입체로 읽힌다.
CAR_BODY = (50, 54, 66)     # 차체(보닛·트렁크)
CAR_ROOF = (82, 88, 106)    # 지붕 — 위에서 빛을 받아 가장 밝다
CAR_GLASS = (24, 27, 38)    # 앞/뒤 유리 — 가장 어둡다
CAR_EDGE = (104, 110, 128)
CAR_LAMP = (232, 226, 190)  # 헤드라이트 — 앞이 어느 쪽인지 알려준다
CAR_TAIL = (196, 72, 68)    # 후미등 — 뒤가 어느 쪽인지


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
        self._prev_arc_deg = None               # 아크 스무딩용 이전 프레임 각도
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
        """3열: 왼쪽 무엇(차종·상태) / 가운데 어디(레이더) / 오른쪽 얼마나(dB)."""
        self._frame = (self._frame + 1) % 100000
        lo = self._lo
        color = vehicle_color(view.sound_text)
        t = hud_card.spl_intensity(view.level_db, view.spl_calibrated)

        surface.blit(self._f_veh.render(view.sound_text, True, FG), lo.veh_xy)
        # 방향은 항상 문구로 낸다. 아크가 방향을 보여주지만, 아크만으로는 '모른다'와
        # '어느 방향이다'의 차이가 약하다. 소리를 못 듣는 운전자에게 이 화면이 유일한
        # 경고라 모르면 모른다고 써야 한다.
        state = view.direction_text if getattr(view, "is_horn", False) \
            else f"{view.direction_text} · {view.motion_text}"
        surface.blit(self._f_state.render(state, True, color), lo.state_xy)

        self._draw_radar(surface, color, t, view.angle_deg, view.direction)
        # spl_calibrated 를 여기서도 본다(viewmodel._level_text 와 이중 방어).
        # 한 곳만 지키면 뷰가 잘못 만들어졌을 때 없는 단위를 지어내 출력한다.
        if view.level_text and view.spl_calibrated:
            self._draw_db(surface, view.level_text, color)
        if view.subtitle:
            lines = hud_card.wrap_text(
                view.subtitle, lambda s: self._f_cap.size(s)[0],
                w - 2 * lo.margin, max_lines=2)
            self._center_multiline(surface, lines, self._f_cap, FG, w // 2, lo.cap_cy)

    def _arc(self, surface, ring, a0, a1, color, width, alpha=None):
        """레이더 링 하나의 호. alpha 를 주면 글로우용으로 반투명하게 겹쳐 그린다."""
        box = self._ring_box(ring)
        a0, a1 = math.radians(a0), math.radians(a1)
        if alpha is None:
            pygame.draw.arc(surface, color, box, a0, a1, width)
            return
        g = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        pygame.draw.arc(g, (*color, alpha), box, a0, a1, width)
        surface.blit(g, (0, 0))

    def _draw_radar(self, surface, color, t, angle_deg, direction):
        """방향 = 아크가 놓이는 각도, 음압 = 몇 링이 차오르는가.

        꺼진 링은 완전한 타원으로 그린다 — 그게 '어디든 올 수 있다'는 척도가 되고,
        그 위에 켜진 아크가 실제 방향을 가리킨다. 4분면으로 스냅하지 않으므로
        후방 우측과 후방 좌측이 화면에서 갈린다.
        """
        lo = self._lo
        levels = hud_card.ring_levels(t, hud_card.RINGS)
        center = hud_card.arc_center_deg(angle_deg, direction)

        # ── 아크 스무딩: 이전 프레임 각도에서 목표로 매 프레임 15%씩 보간 ──
        if center is not None:
            if self._prev_arc_deg is not None:
                center = hud_card.angle_lerp(self._prev_arc_deg, center, 0.15)
            self._prev_arc_deg = center
        else:
            self._prev_arc_deg = None           # 방향 미상이면 보간 리셋

        for r, lv in enumerate(levels):
            self._ring(surface, r, DIM)                       # 꺼진 척도
            if lv <= 0:
                continue
            b = lv * (1.0 - r * 0.10)
            if center is None:
                # 모를 땐 링 전체를 균등하게 약하게 — 한 곳을 단정하지 않는다
                self._ring(surface, r, tuple(int(c * b * 0.42) for c in color),
                           lighting=False)
                continue
            a0, a1 = hud_card.arc_bounds(center)
            w = self._ring_width(r)
            self._arc(surface, r, a0, a1, color, w + 4, alpha=int(52 * b))
            self._arc(surface, r, a0, a1, tuple(int(c * b) for c in color), w)
        self._draw_car(surface)

    def _ring_box(self, ring):
        lo = self._lo
        pad = lo.ring_w // 2 + ring * lo.ring_gap
        return pygame.Rect(lo.radar_cx - lo.radar_rx - pad, lo.radar_cy - lo.radar_ry - pad,
                           (lo.radar_rx + pad) * 2, (lo.radar_ry + pad) * 2)

    def _ring_width(self, ring):
        """바깥 링일수록 얇게 — 멀어질수록 가늘어 보이는 깊이 단서.

        기하(링의 위치·크기)는 건드리지 않는다. 원근으로 눕히면 아래쪽 링들이
        뭉쳐서 '몇 칸 중 몇 칸'이 안 읽히는데, 그게 이 화면의 핵심 정보다.
        두께와 조명만으로 깊이를 주면 척도도 방향 대칭도 그대로다.
        """
        return max(2, self._lo.ring_w - ring)

    def _ring(self, surface, ring, color, lighting=True):
        """링 하나를 완전한 타원으로. 켜진 아크가 놓일 자리를 보여 주는 척도다.

        lighting=True 면 아래쪽 반을 한 번 더 밝게 덧그린다 — 아래에서 빛이 오는
        것처럼 보여 평면이 아니라 접시로 읽힌다.

        ⚠ 방향 미상일 때 켜는 링에는 조명을 주지 않는다. 아래가 밝으면 운전자가
        '후방'으로 읽는데, 그 상태의 뜻은 정확히 '어디인지 모른다' 이다. 조명은
        척도의 깊이 단서지 방향 표시가 아니다.
        """
        box, w = self._ring_box(ring), self._ring_width(ring)
        pygame.draw.ellipse(surface, color, box, w)
        if lighting:
            lit = tuple(min(255, int(c * 1.55) + 6) for c in color)
            pygame.draw.arc(surface, lit, box, math.radians(196), math.radians(344), w)

    def _draw_car(self, surface):
        """위에서 본 차. 지붕(밝음)·유리(어두움)·등(앞 흰색/뒤 빨강)으로 입체와 방향을 준다.

        평평한 회색 덩어리면 무슨 도형인지 모르고, 앞뒤도 안 읽힌다. 명암을 층으로
        쌓으면 작은 크기에서도 '위에서 본 차'로 읽힌다.
        """
        lo = self._lo
        cx, cy = lo.radar_cx, lo.radar_cy
        # 위에서 본 차는 길이:폭이 대략 2:1 이다. 정사각형에 가까우면 승합차로 보인다.
        cw = max(14, int(lo.radar_rx / 3.6))
        ch = max(28, int(lo.radar_ry * 1.18))
        top, bot = cy - ch // 2, cy + ch // 2
        hw = cw / 2.0

        def yy(f):      # 차 앞(0.0)에서 뒤(1.0) 사이 비율 → y
            return top + ch * f

        # 바닥 그림자 — 차가 지면에서 떠 보이게
        sh = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        pygame.draw.ellipse(sh, (0, 0, 0, 110),
                            pygame.Rect(int(cx - hw * 1.25), int(bot - ch * 0.12),
                                        int(hw * 2.5), int(ch * 0.30)))
        surface.blit(sh, (0, 0))

        body = [(cx - hw * 0.62, yy(0.0)), (cx + hw * 0.62, yy(0.0)),
                (cx + hw, yy(0.16)), (cx + hw, yy(0.86)),
                (cx + hw * 0.72, yy(1.0)), (cx - hw * 0.72, yy(1.0)),
                (cx - hw, yy(0.86)), (cx - hw, yy(0.16))]
        pygame.draw.polygon(surface, CAR_BODY, body)

        # 지붕(캐빈) — 가장 밝은 면
        roof = pygame.Rect(int(cx - hw * 0.74), int(yy(0.34)),
                           int(hw * 1.48), int(ch * 0.34))
        pygame.draw.rect(surface, CAR_ROOF, roof, border_radius=max(2, int(hw * 0.22)))

        # 앞/뒤 유리 — 지붕과 보닛/트렁크 사이의 어두운 띠
        for y0, y1 in ((0.26, 0.35), (0.67, 0.76)):
            pygame.draw.polygon(surface, CAR_GLASS, [
                (cx - hw * 0.60, yy(y0)), (cx + hw * 0.60, yy(y0)),
                (cx + hw * 0.74, yy(y1)), (cx - hw * 0.74, yy(y1))])

        # 사이드미러
        for sx in (-1, 1):
            pygame.draw.circle(surface, CAR_EDGE,
                               (int(cx + sx * hw * 1.06), int(yy(0.36))),
                               max(1, int(hw * 0.11)))

        pygame.draw.polygon(surface, CAR_EDGE, body, 2)

        lamp_w, lamp_h = max(2, int(hw * 0.30)), max(2, int(ch * 0.045))
        for sx in (-1, 1):
            pygame.draw.rect(surface, CAR_LAMP,
                             (int(cx + sx * hw * 0.52 - lamp_w / 2), int(yy(0.03)),
                              lamp_w, lamp_h), border_radius=1)
            pygame.draw.rect(surface, CAR_TAIL,
                             (int(cx + sx * hw * 0.46 - lamp_w / 2), int(yy(0.94)),
                              lamp_w, lamp_h), border_radius=1)

    def _draw_db(self, surface, text, color):
        """음압 숫자 — 레이더 오른쪽, 세로 중앙."""
        lo = self._lo
        num = self._f_db.render(text, True, color)
        unit = self._f_unit.render(" dB", True, MUTED)
        x = lo.db_right - num.get_width() - unit.get_width()
        y = lo.radar_cy - num.get_height() // 2
        surface.blit(num, (x, y))
        surface.blit(unit, (x + num.get_width(),
                            y + num.get_height() - unit.get_height() - 4))

    def _draw_normal(self, surface, view, w, h):
        """평상시도 같은 레이더를 쓴다 — 긴급 전환 시 화면이 재배치되지 않는다."""
        self._frame = (self._frame + 1) % 100000
        lo = self._lo
        surface.blit(self._f_veh.render("듣는 중", True, MUTED), lo.veh_xy)
        self._draw_radar(surface, MUTED, 0.0, None, Direction.UNKNOWN)
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
