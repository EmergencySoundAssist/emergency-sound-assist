"""고정 그리드 불변성 — 상태가 바뀌어도 요소가 이동·신축하지 않는다.

운전 중 흘긋 볼 때 눈이 매번 요소를 다시 찾으면 안 된다. v2 는 음압에 따라 칸
높이가 14~58px 로 변하고 전방일 때 텍스트가 중앙으로 순간이동했다. 개편의
목적은 좌표를 고정하는 것 자체이므로, 이 파일이 실패하면 테스트가 아니라
렌더러를 고친다.
"""
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

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
    """frames=3 은 의도적이다 — 늘리지 말 것.

    접근 중 상태는 ripple 이 돌고 주기가 음압마다 다르다(15~28프레임). frames 를
    20/22/24 로 두면 어떤 상태는 하필 위상이 어두운 지점에 걸려 칸 밝기가 임계값
    260 밑으로 내려가고, `len(bands) == 7` 이 "어떤 상태가 통째로 안 켜졌다"며
    실패한다 — 정작 이 파일이 지키려는 불변식(칸 세로 구간 (133,166))은 멀쩡한데
    말이다. 3 은 모든 주기에서 위상이 아직 밝은 구간이라 9개 상태가 같은 조건으로
    비교된다. 밝기가 아니라 '좌표가 안 움직인다'를 재는 파일이므로 이걸로 충분하다.
    """
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
    assert len(bands) == 7, "밝은 칸이 있는 상태 수가 실측(7)과 다르다 — 어떤 상태가 통째로 안 켜졌다"
    spans = set(bands.values())
    assert len(spans) == 1, f"칸 세로 구간이 상태마다 다르다: {bands}"
    assert bands[next(iter(bands))][1] - bands[next(iter(bands))][0] + 1 == lo.seg_h, \
        "잡힌 구간 높이가 Layout.seg_h 와 다르다"


def test_lit_cell_count_does_not_grow_with_loudness():
    """음압이 커져도 '켜진 칸 수'가 같아야 한다 = 클러스터 폭 불변.

    위 테스트는 세로 밴드만 본다 — 삭제된 spread_for_gauge(음압이 클수록 클러스터를
    옆으로 넓히던 것)가 되살아나도 통과해 버린다. 이 개편의 목적 자체가 '기하가
    움직이지 않는다'이므로 가로도 못박는다.

    칸 경계는 픽셀 리터럴이 아니라 _bar_span()(칸 피치의 유일한 계산처)에서 받는다.
    임계값 260 은 위 테스트와 같은 이유다: 글로우 헤일로를 칸으로 잘못 세지 않으면서
    순색 칸(sum>=333)만 잡는다. 헤일로는 음압에 따라 번지는 폭이 달라 140 같은
    값으로는 폭 회귀와 구분되지 않는다.

    칸 '전체'가 아니라 가운데 열만 본다. 실측하면 97dB 에서 이웃 칸의 헤일로가
    옆칸 가장자리를 sum=264 까지 밀어 올려 260 을 넘긴다 — 칸이 아니라 번짐인데
    칸으로 세어 66/79/97 이 1/1/2 로 갈린다(거짓 실패). 가운데 열은 기하학적으로
    안전하다: 이웃 글로우는 pad(<=16px) 만큼만 넘어와 칸 시작+24 까지 닿고,
    가운데는 시작+seg_w//2(=16) 라 8px 여유가 있다. 가운데 열 실측값은
    66/79/97 모두 [.. 216, 336, 216 ..] 로 중심 칸 하나만 260 을 넘는다.
    """
    lo = Layout.for_size(W, H)
    r = Renderer(HudConfig(fullscreen=False))
    x0, _, seg_w, gap = r._bar_span()

    def lit_cells(spl):
        surf = _render(_view(level_db=spl, level_text=f"{spl:.0f}"))
        return sum(
            1 for i in range(lo.seg_n)
            if sum(surf.get_at((x0 + i * (seg_w + gap) + seg_w // 2,
                                lo.bar_cy))[:3]) > 260
        )

    counts = {spl: lit_cells(spl) for spl in (66.0, 79.0, 97.0)}
    assert len(set(counts.values())) == 1, f"음압에 따라 켜진 칸 수가 변한다: {counts}"
    assert all(c > 0 for c in counts.values()), f"어느 음압에서도 칸이 안 켜졌다: {counts}"


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
    assert len(lefts) == 9, "차종 텍스트가 있는 상태 수가 실측(9)과 다르다 — 어떤 상태가 통째로 안 켜졌다"
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

    처음엔 bar_cy ± seg_h 구간을 통째로 검사에서 뺐다 — 그 높이에서 걸리는
    "충돌"이 실은 바 자체의 "L" 눈금 라벨(x=574~583, y=144~157, MUTED 색)이라고
    진단했기 때문이다. 진단은 맞았지만 처방이 과했다: 그 대역은 172행 중 68행
    (40%)이나 되고 차종 텍스트(y 78~202)의 세로 한가운데를 관통한다 — 즉 상태
    텍스트가 bar_cy 높이(y 124~176)로 재배치돼 x=621까지 뻗어도 이 테스트는
    못 본다.

    그래서 행은 하나도 빼지 않고, 대신 x 폭을 L 라벨이 실제로 앉는 자리만큼만
    좁힌다: x0(_bar_span()의 유일한 계산처) 에서 label_gap(=_draw_bar 가 쓰는
    lo.margin // 3, 바로 그 표현식) 만큼 왼쪽까지 ~ bar_x 직전. 1280 기준
    x=584~599. 9개 상태 전부 0건으로 통과함을 확인했다.
    """
    lo = Layout.for_size(W, H)
    x0, _, _, _ = Renderer(HudConfig(fullscreen=False))._bar_span()
    for name, view in STATES.items():
        surf = _render(view)
        rows = range(lo.veh_xy[1], min(H, lo.state_xy[1] + 50))
        gutter = [x for x in range(x0 - lo.margin // 3, lo.bar_x)
                  for y in rows
                  if sum(surf.get_at((x, y))[:3]) > 120]
        assert not gutter, f"{name}: 텍스트가 바 영역까지 넘어왔다 (x={sorted(set(gutter))[:5]})"
