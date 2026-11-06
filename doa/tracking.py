"""DoA 시간적 견고화 — 반사·잡음으로 방향이 한 프레임씩 '튀는' 것을 억제.

순수 로직만 둔다 (표준 라이브러리만 의존 → 하드웨어·pyroomacoustics 없이 단위 테스트 가능).
`multi_live` 실시간 루프가 매 프레임 (대표 방향, 신뢰도)를 넣으면, 최근 창(window)의
'신뢰 프레임'을 **원형 다수결(circular median)** 해 안정된 대표 방향을 돌려준다.

원리:
  · 진짜 음원 방향은 시간상 안정적 → 여러 프레임의 중앙값으로 모인다.
  · 반사/잡음으로 반대편(±180°)으로 튄 한두 프레임은 중앙값에서 밀려난다(아웃보팅).
  · 공간 스펙트럼이 평평한(저신뢰) 프레임은 투표에서 제외 → 모자라면 '방향 불확실'.

검출 지연과 무관: 입력은 '이미 추정된 방향'이라, 검출(소리 왔다)은 즉시 그대로 가고
여기서는 **표시할 방향(LED 화살표)** 만 안정화한다.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Sequence, Tuple


def _ang_diff(a: float, b: float) -> float:
    """두 방위각의 최소 원형 거리(0~180°)."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def circular_median(angles_deg: Sequence[float]) -> Optional[float]:
    """방위각들의 원형 중앙값(도). 비면 None.

    후보를 데이터점으로 두고 '나머지와의 원형거리 합'이 최소인 점을 고른다(O(n²),
    n은 한 창의 프레임 수라 작다). 원형 평균과 달리 ±180°로 튄 소수 이상치에 강건하다.
    예) [10, 12, 8, 190, 11] → 11° 부근 (190°는 아웃보팅).
    """
    a = [float(x) % 360.0 for x in angles_deg]
    if not a:
        return None
    best = min(a, key=lambda c: sum(_ang_diff(c, x) for x in a))
    return float(best % 360.0)


@dataclass
class TrackResult:
    """트래커 한 스텝 결과."""
    angle: Optional[float]   # 안정된 대표 방위각(도). None = 방향 불확실
    n_confident: int         # 현재 창 안의 신뢰 프레임 수
    n_window: int            # 현재 창의 전체 프레임 수


class DirectionTracker:
    """최근 창의 신뢰 프레임을 원형 다수결해 방향을 안정화한다.

    Args:
        maxlen:     창에 보관할 프레임 수 (= round(window_sec / hop_sec)).
        conf_min:   이 신뢰도 미만 프레임은 투표에서 제외.
        min_frames: 창 안 신뢰 프레임이 이 수 이상이어야 방향 확정(미만이면 None).
    """

    def __init__(self, maxlen: int, conf_min: float, min_frames: int) -> None:
        self.buf: Deque[Tuple[Optional[float], float]] = deque(maxlen=max(1, maxlen))
        self.conf_min = conf_min
        self.min_frames = max(1, min_frames)

    def update(self, angle: Optional[float], conf: float) -> TrackResult:
        """한 프레임 반영. 미검출/무음 프레임은 angle=None 으로 넣어 자연히 노후화시킨다."""
        self.buf.append((angle, float(conf)))
        confident: List[float] = [
            a for (a, c) in self.buf if a is not None and c >= self.conf_min
        ]
        if len(confident) >= self.min_frames:
            return TrackResult(circular_median(confident), len(confident), len(self.buf))
        return TrackResult(None, len(confident), len(self.buf))

    def reset(self) -> None:
        self.buf.clear()
