"""HUD LED 카드의 순수 계산 로직 (pygame 의존 없음 → 단위테스트 용이).

방향각/이산방향 → 세그먼트 위치, 접근 빠르기 → blink 주기, 세그먼트 밝기 감쇠.
데이터가 없을 때(angle None / speed None)의 성능저하 규칙도 여기서 결정한다.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from core.types import Direction, Motion


def direction_to_index(angle_deg: Optional[float], direction: Direction, n: int = 15) -> int:
    """점등 중심 세그먼트 인덱스(0=맨좌 ~ n-1=맨우).

    angle_deg 있으면 -90°~+90°를 선형 매핑(0°=중앙). 없으면 이산 방향으로 스냅.
    """
    mid = n // 2
    if angle_deg is not None:
        a = max(-90.0, min(90.0, angle_deg))          # 범위 고정
        return round((a + 90.0) / 180.0 * (n - 1))
    # 이산 폴백
    if direction is Direction.LEFT:
        return n // 4
    if direction is Direction.RIGHT:
        return n - 1 - n // 4
    return mid                                         # FRONT/REAR/UNKNOWN → 중앙


# speed_level 1~5 → blink 주기 프레임(느림 30 ~ 빠름 9), 라벨
_SPEED_LABEL = {5: "빠르게", 4: "빠르게", 3: "보통", 2: "느리게", 1: "느리게"}
_SPEED_PERIOD = {5: 9, 4: 12, 3: 18, 2: 24, 1: 30}


def blink_spec(speed_level: Optional[int], motion: Motion) -> Tuple[str, int]:
    """(라벨, blink 주기프레임). 주기 0 = 상시점등(깜빡임 없음).

    speed_level 있으면 1~5로 정밀 매핑. 없으면 Motion 폴백:
    접근=보통(18) blink, 멀어짐/유지/미상=상시(0).
    """
    if speed_level is not None:
        lv = max(1, min(5, speed_level))
        return (_SPEED_LABEL[lv], _SPEED_PERIOD[lv])
    if motion is Motion.APPROACHING:
        return ("접근 중", 18)
    labels = {Motion.RECEDING: "멀어짐", Motion.STEADY: "유지", Motion.UNKNOWN: "이동 미상"}
    return (labels.get(motion, "이동 미상"), 0)


def is_lit_now(period_frames: int, frame_count: int) -> bool:
    """blink 현재 on/off. 주기 0이면 항상 on. 앞 절반 on, 뒤 절반 off."""
    if period_frames <= 0:
        return True
    return (frame_count % period_frames) < (period_frames / 2.0)


def segment_brightness(seg_index: int, center_index: int, radius: int = 2) -> float:
    """세그먼트 밝기 0.0~1.0. 중심 1.0, 선형 감쇠, 거리==radius(및 그 밖) 0.

    분모를 radius 로 두어 dist==radius 에서 정확히 0 이 된다(경계=꺼짐).
    """
    dist = abs(seg_index - center_index)
    if dist >= radius:
        return 0.0
    return 1.0 - (dist / radius)


def _ellipsize(text: str, measure: Callable[[str], int], max_width: int) -> str:
    """text 가 폭을 넘으면 뒤를 잘라 '…' 를 붙여 폭 안에 맞춘다."""
    if measure(text) <= max_width:
        return text
    ell = "…"
    s = text
    while s and measure(s + ell) > max_width:
        s = s[:-1]
    return (s + ell) if s else ell


def wrap_text(
    text: str,
    measure: Callable[[str], int],
    max_width: int,
    max_lines: int = 2,
) -> List[str]:
    """자막을 max_width(px) 에 맞춰 그리디 줄바꿈.

    한글은 단어 공백이 불규칙하므로 글자 단위로 채운다(공백은 자연히 줄 끝에 남음).
    max_lines 를 넘기면 마지막 줄을 '…' 로 절단한다. 빈 문자열 → [].
    measure: 문자열 → 픽셀폭 (렌더러에선 font.size(s)[0]).
    """
    if not text:
        return []
    if max_width <= 0 or max_lines <= 0:
        return [text]
    lines: List[str] = []
    cur = ""
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if cur == "" or measure(cur + ch) <= max_width:
            cur += ch
            i += 1
        else:
            # 현재 줄이 꽉 참. 마지막 허용 줄이면 남은 전체를 절단해 담고 종료.
            if len(lines) == max_lines - 1:
                return lines + [_ellipsize(text[i - len(cur):], measure, max_width)]
            lines.append(cur)
            cur = ""
    if cur:
        if len(lines) < max_lines:
            lines.append(cur)
        else:
            lines[-1] = _ellipsize(lines[-1] + cur, measure, max_width)
    return lines


def dots_brightness(frame: int, count: int = 3, period: int = 24) -> List[float]:
    """STT 변환 중 로딩 점 애니메이션의 각 점 밝기(0.2~1.0).

    각 점을 period/count 만큼 위상차를 두고 삼각파(0→1→0)로 밝혔다 어둡힌다.
    바닥 밝기 0.2 를 둬 점이 완전히 사라지지 않게 한다(렌더러는 이 밝기로
    색·세로 오프셋을 만들어 '움직이는 점'을 그린다).
    """
    step = period / count
    out: List[float] = []
    for i in range(count):
        t = ((frame - i * step) % period) / period      # 0~1 위상
        tri = 1.0 - abs(2.0 * t - 1.0)                   # 0→1→0 삼각파
        out.append(0.2 + 0.8 * tri)
    return out


def strip_lit(
    is_horn: bool,
    speed_level: Optional[int],
    motion: Motion,
    frame: int,
) -> bool:
    """이번 프레임에 방향 LED 스트립을 켤지 여부.

    경적은 '접근 여부'를 출력하지 않으므로 깜빡이지 않고 상시 점등한다.
    그 외에는 기존 blink 규칙(blink_spec 주기 + is_lit_now)을 따른다.
    """
    if is_horn:
        return True
    _, period = blink_spec(speed_level, motion)
    return is_lit_now(period, frame)


def azimuth_to_bar_index(raw_deg: float, n: int = 15) -> Optional[int]:
    """raw DoA 방위각 → 가로 바 인덱스(0=좌 ~ n-1=우). 전방이면 None(숨김).

    estimator 의 차량 보정(_to_vehicle_angle)을 재사용해 raw(0~359, 보드 기준)를
    차량 기준(전0/우90/후180/좌270)으로 바꾼 뒤 가로 위치로 편다:
      우(90°)→오른쪽 끝, 후(180°)→중앙, 좌(270°)→왼쪽 끝.
    전방(315~360° 및 0~45°)은 운전자가 직접 보는 방향이라 바를 숨긴다(None).
    """
    from doa.estimator import _to_vehicle_angle    # 케이블=후방 보정 재사용(DRY)
    veh = _to_vehicle_angle(raw_deg)               # 0=전 90=우 180=후 270=좌
    if veh >= 315.0 or veh < 45.0:
        return None                                # 전방 → 숨김
    frac = (270.0 - veh) / 180.0                   # 우90→1.0, 후180→0.5, 좌270→0.0
    frac = max(0.0, min(1.0, frac))
    return int(round(frac * (n - 1)))


def direction_visible(direction: Direction) -> bool:
    """긴급 화면에서 이 방향의 바를 표시할지(각도가 없을 때의 이산 폴백). 전방만 숨긴다."""
    return direction is not Direction.FRONT


# ── 접근 빠르기(speed_level) → 퍼짐 반경 · 퍼지는 깜빡임(ripple) ─────────────
RIPPLE_PERIOD = 16          # 퍼짐 깜빡임 한 주기 프레임 수(@30fps). 디자인 확정값.

# speed_level 1~5 → 점등 클러스터 반경(세그먼트). 빠를수록 넓게 = "가까워지는 느낌".
# 접근 중이 아니면 speed_level=None → 기본 반경 3. (디자인 스튜디오에서 확정한 매핑)
_SPEED_SPREAD = {1: 2, 2: 3, 3: 4, 4: 5, 5: 6}


def spread_for_speed(speed_level: Optional[int]) -> int:
    """접근 빠르기(1~5) → 퍼짐 반경. None(접근 아님)이면 기본 3."""
    if speed_level is None:
        return 3
    return _SPEED_SPREAD.get(max(1, min(5, speed_level)), 3)


def should_ripple(is_horn: bool, motion: Motion) -> bool:
    """이 상태에서 '퍼지는 깜빡임'을 줄지. 경적은 상시(퍼짐 X), 접근 중일 때만 퍼진다."""
    return (not is_horn) and (motion is Motion.APPROACHING)


def ripple_brightness(
    seg_index: int, center_index: int, radius: float,
    frame: int, period: int = RIPPLE_PERIOD,
) -> float:
    """퍼지는 깜빡임의 세그먼트 밝기(0~1).

    중앙에서 바깥으로 반경이 1→radius 로 커지며(퍼짐) 커질수록 흐려지다가 리셋 반복.
    """
    ph = (frame % period) / period                 # 0~1 위상
    er = 1.0 + (radius - 1.0) * ph                  # 확장 반경
    return segment_brightness(seg_index, center_index, er) * (1.0 - 0.55 * ph)
