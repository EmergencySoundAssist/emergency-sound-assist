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






# ---------------------------------------------------------------------------
# 레이더 불변성 (v4) — 상태가 바뀌어도 레이더와 차 실루엣이 같은 자리
# ---------------------------------------------------------------------------

def _car_box(surf, lo, thresh=110):
    """중앙 차 실루엣이 차지하는 사각형.

    스캔 창을 차 크기(렌더러와 같은 식)로 좁혀야 한다 — 넓게 잡으면 전방·후방의
    가장 안쪽 아크(중심에서 radar_ry 만큼 위/아래)까지 물어서 상태마다 달라진다.
    """
    cw = max(12, lo.radar_rx // 3)
    ch = max(20, lo.radar_ry * 6 // 7)
    xs, ys = [], []
    for y in range(lo.radar_cy - ch // 2 - 4, lo.radar_cy + ch // 2 + 4):
        for x in range(lo.radar_cx - cw, lo.radar_cx + cw):
            if sum(surf.get_at((x, y))[:3]) > thresh:
                xs.append(x); ys.append(y)
    return (min(xs), min(ys), max(xs), max(ys)) if xs else None


def test_car_silhouette_never_moves_between_states():
    """레이더의 기준점. 이게 흔들리면 아크 위치도 같이 흔들린다."""
    lo = Layout.for_size(W, H)
    boxes = {n: _car_box(_render(v), lo) for n, v in STATES.items()}
    found = {n: b for n, b in boxes.items() if b is not None}
    assert len(found) == len(STATES), f"차 실루엣이 안 보이는 상태가 있다: " \
        f"{[n for n, b in boxes.items() if b is None]}"
    assert len(set(found.values())) == 1, f"차 실루엣이 상태마다 다르다: {found}"


def test_radar_reach_is_identical_across_states():
    """켜진 링 수는 음압 따라 달라도, 꺼진 링까지 포함한 '척도' 전체는 안 변한다."""
    from hud.card import RINGS
    lo = Layout.for_size(W, H)
    # 스캔을 레이더 밴드로 한정한다 — 화면 전체를 훑으면 같은 x 열에 있는 자막
    # 글자까지 물어서 '자막동반' 상태만 범위가 달라진다.
    reach = lo.radar_ry + lo.ring_w // 2 + (RINGS - 1) * lo.ring_gap
    top, bot = max(0, lo.radar_cy - reach - 6), min(H, lo.radar_cy + reach + 6)
    spans = {}
    for name, view in STATES.items():
        surf = _render(view)
        ys = [y for y in range(top, bot)
              if any(sum(surf.get_at((x, y))[:3]) > 60
                     for x in range(lo.radar_cx - 4, lo.radar_cx + 4))]
        if ys:
            spans[name] = (min(ys), max(ys))
    assert len(spans) >= 8, f"세로로 레이더가 잡힌 상태가 적다 ({len(spans)})"
    assert len(set(spans.values())) == 1, f"레이더 세로 범위가 상태마다 다르다: {spans}"


def test_radar_never_overlaps_the_caption_row():
    """자막·dots 가 아크 위에 겹쳐 그려지면 둘 다 읽기 나빠진다."""
    from hud.card import RINGS
    lo = Layout.for_size(W, H)
    bottom = lo.radar_cy + lo.radar_ry + lo.ring_w // 2 + (RINGS - 1) * lo.ring_gap
    assert bottom < lo.cap_cy, f"레이더 하단 {bottom} 이 자막 줄 {lo.cap_cy} 를 침범한다"
