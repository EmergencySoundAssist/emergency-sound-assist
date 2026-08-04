"""FusedResult → HudView 순수 변환.

렌더러가 화면을 그릴 때 필요한 '표시용 값'만 담은 얇은 뷰모델. pygame 의존이 없어
단위 테스트가 쉽다. 문구는 core.types.FusedResult.to_korean 과 동일한 한국어를 쓴다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.types import (
    FusedResult, SoundClass, SirenSubtype, Direction, Motion,
)

_KO_SUBTYPE = {
    SirenSubtype.AMBULANCE: "구급차",
    SirenSubtype.POLICE: "경찰차",
    SirenSubtype.FIRE: "소방차",
    SirenSubtype.UNKNOWN: "긴급차량",
}
_KO_SOUND = {
    SoundClass.SIREN: "사이렌",
    SoundClass.HORN: "경적",
    SoundClass.NORMAL_TRAFFIC: "일반 도로 소음",
}
_KO_DIR = {
    Direction.FRONT: "전방",
    Direction.REAR: "후방",
    Direction.LEFT: "좌측",
    Direction.RIGHT: "우측",
    Direction.UNKNOWN: "방향 미상",
}
_KO_MOTION = {
    Motion.APPROACHING: "접근 중",
    Motion.RECEDING: "멀어짐",
    Motion.STEADY: "유지",
    Motion.UNKNOWN: "이동 미상",
}

# 게이지(시작 음량 대비 상승분) → 접근 중에 쓸 거리 문구의 경계.
_GAUGE_NEAR = 0.75
_GAUGE_MID = 0.35


def _proximity_text(label, gauge, motion) -> Optional[str]:
    """화면에 띄울 거리 문구.

    detector 의 proximity 는 '이벤트 최고 음량(=최근접) 대비' 값이라, 최근접을
    지나기 전에는 기준점이 없어 None 이다. 그런데 거리가 가장 궁금한 구간이
    바로 다가오는 동안이다. 그 구간에서는 게이지(시작 음량 대비 상승분)로
    문구를 만든다 — 기준점만 다를 뿐 둘 다 '얼마나 가까워졌나'를 잰다.

    다만 '최근접'은 최고점이 확정된 뒤에만 쓴다. 접근 중에는 아무리 가까워도
    '근접'까지만 말한다 — 아직 더 가까워질 수 있기 때문이다.

    ⚠ 어느 쪽도 미터가 아니다. 마이크 하나로는 음원의 절대 음압을 모르므로
    절대 거리를 낼 수 없다(docs/approach/design.md).
    """
    if label is not None:
        return label                      # 최근접을 지난 뒤 — 최고점 대비 값
    if gauge is None or motion is not Motion.APPROACHING:
        return None
    if gauge >= _GAUGE_NEAR:
        return "근접"
    return "근거리" if gauge >= _GAUGE_MID else "원거리"


@dataclass
class HudView:
    """렌더러가 소비하는 표시용 뷰모델."""
    emergency: bool
    sound_text: str
    direction: Direction
    direction_text: str
    motion_text: str
    subtitle: str
    confidence: float
    angle_deg: Optional[float] = None       # 연속 방향각(-90좌~+90우). 없으면 None
    speed_level: Optional[int] = None       # 접근 빠르기 1~5. 없으면 None
    motion: Motion = Motion.UNKNOWN         # 원본 Motion(렌더러 blink 판단용)
    is_horn: bool = False                   # 경적이면 True → 접근/이동 출력 안 함
    gauge: Optional[float] = None           # 근접 게이지 0~1 (거리감) → 바 퍼짐 폭·속도
    # 상대 근접도 라벨. '최근접/근거리/원거리' — 절대 거리(m)가 아니라 이벤트 내
    # 가장 컸던 순간 대비 위치다. 미터로 읽히지 않게 화면에서도 그대로 쓴다.
    proximity: Optional[str] = None
    # 최근접 대비 거리비(≥1.0). 최고점을 지난 뒤에만 값이 있다 — 기준점이 그때
    # 확정되기 때문. 접근 중에는 None 이고, 이것도 미터가 아니다.
    rel_distance: Optional[float] = None

    def approach_motion(self) -> Motion:
        """렌더러가 blink 판단에 쓰는 원본 Motion 접근자."""
        return self.motion

    @classmethod
    def from_fused(cls, fused: FusedResult) -> "HudView":
        s = fused.sound
        if s.label is SoundClass.SIREN and s.subtype is not None:
            sound_text = _KO_SUBTYPE[s.subtype]     # 사이렌 → 차종
        else:
            sound_text = _KO_SOUND[s.label]
        d = fused.direction.direction
        sp = fused.speech
        subtitle = sp.text if (sp is not None and sp.is_speech and sp.text) else ""
        angle_deg = fused.direction.angle_deg
        speed_level = getattr(fused.approach, "speed_level", None)
        gauge = getattr(fused.approach, "gauge", None)
        proximity = _proximity_text(
            getattr(fused.approach, "proximity", None), gauge, fused.approach.motion)
        is_horn = s.label is SoundClass.HORN
        return cls(
            emergency=s.is_emergency,
            sound_text=sound_text,
            direction=d,
            direction_text=_KO_DIR[d],
            motion_text=_KO_MOTION[fused.approach.motion],
            subtitle=subtitle,
            confidence=s.confidence,
            angle_deg=angle_deg,
            speed_level=speed_level,
            motion=fused.approach.motion,
            is_horn=is_horn,
            gauge=gauge,
            proximity=proximity,
            rel_distance=getattr(fused.approach, "rel_distance", None),
        )
