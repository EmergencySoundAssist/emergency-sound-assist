"""고정 그리드 불변성 — 상태가 바뀌어도 요소가 이동·신축하지 않는다.

운전 중 흘긋 볼 때 눈이 매번 요소를 다시 찾으면 안 된다. v2 는 음압에 따라 칸
높이가 14~58px 로 변하고 전방일 때 텍스트가 중앙으로 순간이동했다. 개편의
목적은 좌표를 고정하는 것 자체이므로, 이 파일이 실패하면 테스트가 아니라
렌더러를 고친다.
"""
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
import pytest

from core.types import Direction, Motion
from hud.card import Layout
from hud.config import HudConfig
from hud.renderer import Renderer
from hud.viewmodel import HudView

W, H = 1280, 360


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
    "우측_큰소리": _view(),
    "좌측_작은소리": _view(direction=Direction.LEFT, direction_text="좌측",
                          level_db=66.0, level_text="66"),
    "후방_중간": _view(direction=Direction.REAR, direction_text="후방",
                       level_db=79.0, level_text="79"),
    "멀어짐": _view(motion=Motion.RECEDING, motion_text="멀어짐"),
    "전방": _view(direction=Direction.FRONT, direction_text="전방"),
    "경적": _view(sound_text="경적", is_horn=True, motion=Motion.UNKNOWN,
                  motion_text="이동 미상"),
    "미보정": _view(level_text=None, spl_calibrated=False),
    "자막동반": _view(subtitle="구급차가 지나갑니다"),
    "평상시": _view(emergency=False, sound_text="일반 도로 소음",
                    motion=Motion.UNKNOWN, motion_text="이동 미상"),
}


def _render(view, frames=3):
    pygame.init()
    surf = pygame.Surface((W, H))
    r = Renderer(HudConfig(fullscreen=False))
    for _ in range(frames):
        r.draw(surf, view)
    return surf


def _lit_rows(surf, x0, x1, y0, y1, thresh):
    """구간 안에서 밝은 픽셀이 있는 y 좌표 집합."""
    return {y for y in range(y0, y1)
            for x in range(x0, x1)
            if sum(surf.get_at((x, y))[:3]) > thresh}


def test_bar_row_band_is_identical_across_every_state():
    """칸이 차지하는 세로 구간이 모든 상태에서 같아야 한다 = 칸 높이 불변.

    임계값은 140 이 아니라 260 을 쓴다. 실측(9개 상태 렌더 후 픽셀 덤프)해 보면
    글로우 헤일로가 상태(음압→glow 배수)에 따라 최대 sum=225 까지 번지고, DIM
    빈 칸은 sum=108 이다. 140 은 이 둘 사이 어디에도 안전하지 않아 헤일로를
    칸으로 잘못 세고, 상태마다 번지는 폭이 달라 (117,182)/(133,166)/(116,183)
    세 가지 다른 구간이 나와 버린다 — 그리드는 안 움직였는데 테스트만 흔들리는
    거짓 실패다. 실제 칸의 순색(다 켜진 세그먼트)은 모든 상태·색에서 sum>=333
    이었으므로 225~333 사이인 260 을 쓰면 헤일로를 배제하고 칸 자체만 잡는다.
    """
    lo = Layout.for_size(W, H)
    top, bot = lo.bar_cy - lo.seg_h, lo.bar_cy + lo.seg_h
    bands = {}
    for name, view in STATES.items():
        surf = _render(view)
        rows = _lit_rows(surf, lo.bar_x, lo.bar_x + lo.bar_w, top, bot, thresh=260)
        if rows:
            bands[name] = (min(rows), max(rows))
    assert len(bands) >= 6, "밝은 칸이 있는 상태가 너무 적다 — 테스트가 무의미하다"
    spans = set(bands.values())
    assert len(spans) == 1, f"칸 세로 구간이 상태마다 다르다: {bands}"
    assert bands[next(iter(bands))][1] - bands[next(iter(bands))][0] + 1 == lo.seg_h, \
        "잡힌 구간 높이가 Layout.seg_h 와 다르다"


def test_vehicle_text_starts_at_the_same_x_in_every_state():
    """차종 텍스트가 상태에 따라 왼쪽↔중앙으로 튀면 안 된다."""
    lo = Layout.for_size(W, H)
    lefts = {}
    for name, view in STATES.items():
        surf = _render(view)
        xs = [x for x in range(0, lo.bar_x)
              for y in range(lo.veh_xy[1], lo.veh_xy[1] + 90)
              if sum(surf.get_at((x, y))[:3]) > 120]
        if xs:
            lefts[name] = min(xs)
    assert len(lefts) >= 6
    assert max(lefts.values()) - min(lefts.values()) <= 6, f"차종 x 가 흔들린다: {lefts}"


def test_meter_track_occupies_the_same_place_in_every_state():
    """미터 트랙(빈 트랙=DIM sum108, 채워진 부분=차량색 sum361~437) 시작 x 불변.

    기대값은 Layout 픽셀 숫자를 여기서 다시 계산하지 않고 Renderer._bar_span()
    (칸 피치의 유일한 계산처) 에서 그대로 받는다 — 두 번째로 계산하면 그 계산이
    렌더러와 몰래 어긋나도 테스트가 자기 자신과만 비교해 통과해버린다.
    임계값 60 은 BG(sum31) 보다는 크고 DIM(sum108) 보다는 작다 — 빈 트랙도
    "트랙이 있다"로 잡아야 하는 것이지(채워진 정도를 재는 게 아니다), 108 미만을
    금지하는 규칙은 '채워진 길이'를 잴 때만 해당한다(다른 테스트가 겪은 버그).
    """
    lo = Layout.for_size(W, H)
    x0, _, _, _ = Renderer(HudConfig(fullscreen=False))._bar_span()
    for name, view in STATES.items():
        surf = _render(view)
        row = lo.meter_y + lo.meter_h // 2
        xs = [x for x in range(lo.bar_x - 40, W)
              if sum(surf.get_at((x, row))[:3]) > 60]
        assert xs, f"{name}: 미터 트랙이 없다"
        assert abs(min(xs) - x0) <= 4, f"{name}: 트랙 시작이 다르다"


def test_state_text_never_collides_with_the_bar_column():
    """왼쪽 텍스트가 오른쪽 바 영역을 침범하면 두 블록이 겹쳐 읽힌다.

    bar_cy ± seg_h(=test 1 과 같은 값) 구간은 검사에서 뺀다 — 그 자리엔 바
    자체의 "L" 눈금 라벨이 x0 바로 옆(gutter 안쪽)에 항상 그려진다(디자인
    의도). 처음엔 이 구간을 포함해서 스캔했다가 모든 상태에서 "충돌"이
    잡혔는데, 실측해 보니 x=575~579·y=144~157 픽셀이 상태 텍스트가 아니라
    이 L 라벨 자체였다(색이 MUTED 계열 (71,71,74)). 왼쪽 텍스트 침범이 아니라
    바 위젯의 정상 구성요소를 오검출한 것이므로 제외한다.
    """
    lo = Layout.for_size(W, H)
    bar_top, bar_bot = lo.bar_cy - lo.seg_h, lo.bar_cy + lo.seg_h
    for name, view in STATES.items():
        surf = _render(view)
        rows = [y for y in range(lo.veh_xy[1], min(H, lo.state_xy[1] + 50))
                if not (bar_top <= y < bar_bot)]
        gutter = [x for x in range(lo.bar_x - 30, lo.bar_x)
                  for y in rows
                  if sum(surf.get_at((x, y))[:3]) > 120]
        assert not gutter, f"{name}: 텍스트가 바 영역까지 넘어왔다 (x={sorted(set(gutter))[:5]})"
