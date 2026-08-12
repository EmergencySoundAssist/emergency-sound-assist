"""HUD LED 카드의 순수 계산 로직 (pygame 의존 없음 → 단위테스트 용이).

방향각/이산방향 → 세그먼트 위치, 음압 강도 → 미터·글로우·퍼지는 깜빡임(ripple)
주기, 세그먼트 밝기 감쇠. 데이터가 없을 때(angle None)의 폴백 규칙도 여기서 결정한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from core.types import Direction, Motion






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






# ── 고정 레이아웃 ────────────────────────────────────────────────────────
# 좌표는 여기에서만 나온다. 상태가 바뀌어도 요소가 이동·신축하지 않아야 운전 중
# 흘긋 볼 때 눈이 매번 같은 자리를 본다. 비율은 1280×360(윈드실드 띠)에서 뽑았다.
@dataclass(frozen=True)
class Layout:
    """HUD 고정 그리드 좌표. (w, h) 에서 한 번 계산하고 이후 불변."""
    margin: int
    veh_xy: Tuple[int, int]
    state_xy: Tuple[int, int]
    db_right: int
    cap_cy: int
    # 폰트 크기도 여기서 나온다. 좌표만 모으고 폰트를 렌더러에 남기면 둘이 따로
    # 놀아 그리드가 어긋난다(차종 104px 자리에 40px 글자가 앉는 식).
    f_veh: int
    f_state: int
    f_db: int
    f_unit: int
    f_cap: int
    # 레이더(v4) — 중심·반지름·링 간격. 바/미터 좌표를 대체한다.
    radar_cx: int
    radar_cy: int
    radar_rx: int
    radar_ry: int
    ring_gap: int
    ring_w: int

    @classmethod
    def for_size(cls, w: int, h: int) -> "Layout":
        margin = round(w * 0.05625)
        # f_veh 는 h 로, 텍스트 칸 폭은 w 로 자란다 — 세로가 긴 화면에서 차종 글자가
        # 레이더 영역을 침범한다(1280×720 의 "구급차", 800×480 의 모든 문구). 레이더
        # 왼쪽 끝까지의 텍스트 칸에 맞춰 상한을 건다. 나누는 수 4.0 은 최장 문구
        # "긴급차량"의 폭÷폰트크기(≈3.48)에 여백을 더한 값이다.
        # 1280×360 에선 427/4.0≈107 > 104 라 기준 해상도 값은 그대로 104.
        radar_cx = round(w * 0.54688)
        radar_rx = round(w * 0.11719)
        ring_gap = max(5, round(h * 0.03333))
        ring_w = max(3, round(h * 0.01667))
        radar_left = radar_cx - radar_rx - ring_w // 2 - (RINGS - 1) * ring_gap
        f_veh = max(24, min(round(h * 0.28889), round((radar_left - margin) / 4.0)))
        return cls(
            margin=margin,
            veh_xy=(margin, round(h * 0.21667)),
            state_xy=(margin + round(w * 0.003), round(h * 0.55556)),
            db_right=w - margin,
            cap_cy=round(h * 0.89444),
            f_veh=f_veh,
            f_state=max(14, round(h * 0.12222)),
            f_db=max(12, round(h * 0.08889)),
            f_unit=max(10, round(h * 0.05278)),
            f_cap=max(14, round(h * 0.10556)),
            # 레이더는 자막 줄(cap_cy) 위에서 끝나야 한다. 아래 값들은
            # radar_cy + radar_ry + ring_w/2 + (RINGS-1)*ring_gap < cap_cy 를 만족한다.
            radar_cx=radar_cx,
            radar_cy=round(h * 0.43889),
            radar_rx=radar_rx,
            radar_ry=round(h * 0.20000),
            ring_gap=ring_gap,
            ring_w=ring_w,
        )


# ── 레이더 (v4) — 방향은 아크 위치, 음압은 채워지는 링 수 ───────────────────
# 가로 막대는 좌우 축만 표현할 수 있어 전/후를 담을 자리가 없었다(전방은 바를 숨기고
# 후방은 중앙으로 밀어넣는 식). 네 방향 아크를 두면 각자 자리를 갖는다.
RINGS = 5                       # 음압 척도. 안쪽에서 바깥으로 찬다.

# ⚠ 이 화면에는 지금 점멸(blink/ripple)이 없다 — 아크는 각도만 바뀌고 밝기는 음압을
# 따라 천천히 변한다. 나중에 맥동·점멸을 넣게 되면 **초당 3회를 넘기지 말 것**:
# 그 이상은 광과민성 발작을 유발할 수 있다(WCAG 2.3.1). 30fps 기준 한 주기 11프레임
# (≈2.7Hz)이 이전 구현이 쓰던 하한이었다.

# 화면 기준 각도(pygame: 0=오른쪽, 반시계 증가). 위=전방.
# 방향은 4분면으로 스냅하지 않는다 — DoA 는 연속 방위각을 주고, 그걸 버리면
# "후방인데 좌측인지 우측인지" 를 알 수 없다. 아크를 측정 각도에 직접 놓는다.
ARC_HALF_DEG = 35.0             # 점등 아크의 반폭

# 각도가 아예 없을 때(1채널 등)의 이산 폴백 — 사분면 중심.
_QUADRANT_SCREEN_DEG = {
    Direction.FRONT: 90.0,
    Direction.RIGHT: 0.0,
    Direction.REAR: 270.0,
    Direction.LEFT: 180.0,
}


def ring_levels(t: float, rings: int = RINGS) -> List[float]:
    """음압 강도 t(0~1) → 링별 밝기 0~1. 안쪽 링이 먼저 찬다.

    칸 수만 쓰면 단계가 rings 개뿐이라 79dB 와 88dB 가 똑같이 3칸으로 뭉친다
    (9dB 차이는 소리 세기로 약 8배다). 칸 수를 척도로 두고 **마지막 칸의 밝기**로
    그 사이를 메운다 — 볼륨 바의 마지막 칸이 반쯤 차는 것과 같은 원리다.
    부분 칸에 바닥 0.25 를 둬서 '켜지다 만' 상태가 눈에 보이게 한다.

    꺼진 링(0.0)도 렌더러가 흐리게 그려야 '몇 칸 중 몇 칸'이라는 척도가 읽힌다.
    """
    if t <= 0:
        return [0.0] * rings
    x = max(t, 0.5 / rings) * rings         # 아주 작은 음압도 첫 칸은 보이게
    out: List[float] = []
    for r in range(rings):
        if r + 1 <= x:
            out.append(1.0)
        elif r < x:
            out.append(0.25 + 0.75 * (x - r))
        else:
            out.append(0.0)
    return out


def arc_center_deg(angle_deg: Optional[float], direction: Direction) -> Optional[float]:
    """점등 아크의 중심각(화면 기준). 방향을 모르면 None.

    angle_deg 가 있으면 그대로 쓴다. 4분면으로 스냅하면 "후방에서 오는데 좌측인지
    우측인지" 를 알 수 없게 되는데, 소리를 못 듣는 운전자에게는 어느 쪽 거울을
    볼지가 바로 그 정보다.

    화면 각도 = 90 - 차량각 (차량각: 전0/우90/후180/좌270 → 화면: 위/오른쪽/아래/왼쪽).
    """
    if angle_deg is not None:
        from doa.estimator import _to_vehicle_angle   # 장착 보정 재사용(DRY)
        return (90.0 - _to_vehicle_angle(angle_deg)) % 360.0
    return _QUADRANT_SCREEN_DEG.get(direction)


def arc_bounds(center_deg: float, half: float = ARC_HALF_DEG):
    """중심각 → (시작각, 끝각). 렌더러가 pygame.draw.arc 에 그대로 넘긴다."""
    return center_deg - half, center_deg + half


# ── 수집 모드 오버레이 (--collect) — 라벨 버튼 띠 ──────────────────────────
# 앞 4개는 tools/tag_siren.py 의 키 배열(1 구급 · 2 경찰 · 3 소방 · u 모름)과 같고,
# not_siren(N)은 오검출 확인용 — 검출이 울렸는데 사이렌이 아니었을 때 누른다
# (검출기 hard-negative 수집 + 미라벨 클립이 다음 라벨을 흡수하는 것 방지).
COLLECT_ORDER = ("ambulance", "police", "fire", "unknown", "horn", "not_siren")


def collect_button_rects(w: int, h: int) -> List[Tuple[int, int, int, int]]:
    """수집 모드 라벨 버튼들의 (x, y, w, h). 화면 하단 한 줄, 터치 가능한 크기.

    좌표 계산을 렌더러 밖에 두는 이유: display 가 터치/클릭 판정에 같은 사각형을
    써야 한다. 렌더러가 그린 곳과 손가락이 닿는 곳이 한 함수에서 나오게 한다.
    """
    n = len(COLLECT_ORDER)
    margin = round(w * 0.05625)                       # Layout.margin 과 동일 비율
    bh = max(34, round(h * 0.16))
    gap = max(6, round(w * 0.008))
    bw = (w - 2 * margin - (n - 1) * gap) // n
    y = h - bh - max(4, round(h * 0.02))
    return [(margin + i * (bw + gap), y, bw, bh) for i in range(n)]


def angle_lerp(prev: float, target: float, alpha: float) -> float:
    """360° 원 위에서 최단 경로 선형 보간.

    prev·target 은 0~360 화면 각도. alpha(0~1)가 클수록 빠르게 쫓는다.
    350° → 10° 를 340° 역회전이 아니라 20° 순회전으로 돌아야 HUD 에서
    방향 전환이 자연스럽다. 차(delta)를 -180~+180 범위로 정규화해 해결한다.
    """
    delta = (target - prev) % 360.0
    if delta > 180.0:
        delta -= 360.0          # 짧은 쪽으로
    return (prev + alpha * delta) % 360.0
