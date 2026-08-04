"""HUD LED 카드의 순수 계산 로직 (pygame 의존 없음 → 단위테스트 용이).

방향각/이산방향 → 세그먼트 위치, 음압 강도 → 미터·글로우·퍼지는 깜빡임(ripple)
주기, 세그먼트 밝기 감쇠. 데이터가 없을 때(angle None)의 폴백 규칙도 여기서 결정한다.
"""
from __future__ import annotations

from dataclasses import dataclass
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


# ── 퍼지는 깜빡임(ripple) 상수 · should_ripple ──────────────────────────────
RIPPLE_PERIOD_FAR = 28      # 멀 때 한 주기 프레임(@30fps) ≈ 0.93초 → 0.9 Hz
RIPPLE_PERIOD_NEAR = 11     # 최근접 ≈ 0.37초 → 2.7 Hz


def should_ripple(is_horn: bool, motion: Motion) -> bool:
    """이 상태에서 '퍼지는 깜빡임'을 줄지. 경적은 상시(퍼짐 X), 접근 중일 때만 퍼진다."""
    return (not is_horn) and (motion is Motion.APPROACHING)


def ripple_brightness(
    seg_index: int, center_index: int, radius: float,
    frame: int, period: int = RIPPLE_PERIOD_FAR,
) -> float:
    """퍼지는 깜빡임의 세그먼트 밝기(0~1).

    중앙에서 바깥으로 반경이 1→radius 로 커지며(퍼짐) 커질수록 흐려지다가 리셋 반복.
    """
    ph = (frame % period) / period                 # 0~1 위상
    er = 1.0 + (radius - 1.0) * ph                  # 확장 반경
    return segment_brightness(seg_index, center_index, er) * (1.0 - 0.55 * ph)


# ── 음압(dB) → 시각량 ────────────────────────────────────────────────────
# 보정 여부에 따라 눈금이 다르다. dBFS 는 항상 음수(0=풀스케일)라, dB SPL 범위를
# 그대로 쓰면 미터가 늘 0 에 붙어 아무 정보도 주지 못한다.
SPL_RANGE = (55.0, 110.0)      # 보정됨(dB SPL): 조용한 실내 ~ 코앞 사이렌
DBFS_RANGE = (-55.0, 0.0)      # 미보정(dBFS)  : 풀스케일 기준

# ponytail: 두 범위 모두 어림값이다. 실차 녹음으로 사이렌의 실제 도달 분포를 본 뒤
# 조정한다 — 도심에서 미터가 늘 절반 이상 차 있으면 눈금을 다시 잡아야 한다.


def spl_intensity(level_db: Optional[float], calibrated: bool) -> float:
    """음압 → 0~1 강도. 값이 없으면 0.0.

    미터 채움·글로우·퍼짐 주기가 전부 이 하나에서 나온다. 화면의 모든 움직임이
    같은 물리량을 가리키게 하려는 것이다(v2 는 폭·속도·색이 서로 다른 값에 묶여
    있었다).
    """
    if level_db is None:
        return 0.0
    lo, hi = SPL_RANGE if calibrated else DBFS_RANGE
    return max(0.0, min(1.0, (level_db - lo) / (hi - lo)))


def spl_to_glow(t: float) -> float:
    """강도 → 글로우 배수.

    상한을 낮게 잡는다. 세게 주면 켜진 칸들이 한 덩어리로 뭉쳐서, 정작 방향
    해상도가 음압에 반비례해 떨어지는 역설이 생긴다(v1 목업에서 확인).
    """
    return 0.5 + 1.1 * max(0.0, min(1.0, t))


def ripple_period_for_spl(t: float) -> int:
    """강도 → 퍼짐 한 주기 프레임 수. 소리가 클수록 짧다(=빨리 퍼진다).

    ⚠ 하한 RIPPLE_PERIOD_NEAR(11프레임 ≈2.7Hz)는 광과민성 안전선이다. 초당 3회를
    넘는 점멸은 발작을 유발할 수 있어(WCAG 2.3.1) 음압이 아무리 커도 내려가지
    않는다. 기존 ripple_period_for_gauge 의 제약을 그대로 승계한다.
    """
    t = max(0.0, min(1.0, t))
    return int(round(RIPPLE_PERIOD_FAR - t * (RIPPLE_PERIOD_FAR - RIPPLE_PERIOD_NEAR)))


# ── 고정 레이아웃 ────────────────────────────────────────────────────────
# 좌표는 여기에서만 나온다. 상태가 바뀌어도 요소가 이동·신축하지 않아야 운전 중
# 흘긋 볼 때 눈이 매번 같은 자리를 본다. 비율은 1280×360(윈드실드 띠)에서 뽑았다.
@dataclass(frozen=True)
class Layout:
    """HUD 고정 그리드 좌표. (w, h) 에서 한 번 계산하고 이후 불변."""
    margin: int
    veh_xy: Tuple[int, int]
    state_xy: Tuple[int, int]
    bar_x: int
    bar_w: int
    bar_cy: int
    seg_h: int
    seg_n: int
    meter_y: int
    meter_h: int
    db_right: int
    db_y: int
    cap_cy: int
    # 폰트 크기도 여기서 나온다. 좌표만 모으고 폰트를 렌더러에 남기면 둘이 따로
    # 놀아 그리드가 어긋난다(차종 104px 자리에 40px 글자가 앉는 식).
    f_veh: int
    f_state: int
    f_db: int
    f_unit: int
    f_cap: int

    @classmethod
    def for_size(cls, w: int, h: int) -> "Layout":
        margin = round(w * 0.05625)
        bar_x = round(w * 0.46875)
        return cls(
            margin=margin,
            veh_xy=(margin, round(h * 0.21667)),
            state_xy=(margin + round(w * 0.003), round(h * 0.55556)),
            bar_x=bar_x,
            bar_w=w - bar_x - margin,
            bar_cy=round(h * 0.41667),
            seg_h=round(h * 0.09444),
            seg_n=15,
            meter_y=round(h * 0.64444),
            meter_h=max(4, round(h * 0.03333)),
            db_right=w - margin,
            db_y=round(h * 0.74444),
            cap_cy=round(h * 0.89444),
            f_veh=max(24, round(h * 0.28889)),
            f_state=max(14, round(h * 0.12222)),
            f_db=max(12, round(h * 0.08889)),
            f_unit=max(10, round(h * 0.05278)),
            f_cap=max(14, round(h * 0.10556)),
        )
