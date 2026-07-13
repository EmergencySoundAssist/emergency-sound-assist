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
        )
