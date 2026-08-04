"""FusedResult → HudView 순수 변환.

렌더러가 화면을 그릴 때 필요한 '표시용 값'만 담은 얇은 뷰모델. pygame 의존이 없어
단위 테스트가 쉽다. 문구는 core.types.FusedResult.to_korean 과 동일한 한국어를 쓴다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from approach.detector import SPL_CALIBRATED
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


def _level_text(level_db, calibrated) -> Optional[str]:
    """화면에 낼 음압 숫자. 미보정이면 None.

    미보정 dBFS 는 운전자에게 의미 없는 숫자다(-4가 큰지 작은지 알 수 없다).
    'dB' 로만 쓰면 물리 음압으로 읽히니 단위를 속이는 셈이 된다. 미보정에서도
    미터·글로우는 상대량으로 같은 정보를 전달하므로 숫자만 빼면 된다.
    """
    if level_db is None or not calibrated:
        return None
    return f"{level_db:.0f}"


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
    motion: Motion = Motion.UNKNOWN         # 원본 Motion(렌더러 blink 판단용)
    is_horn: bool = False                   # 경적이면 True → 접근/이동 출력 안 함
    # 사이렌 대역 레벨(dBFS 또는 보정 시 dB SPL). 이벤트 이력과 무관하게 지금
    # 도달한 소리의 세기라, 접근 초반부터 끊김 없이 값이 있다.
    level_db: Optional[float] = None
    level_text: Optional[str] = None        # 화면 문구 — 단위까지 확정된 형태
    spl_calibrated: bool = False            # 보정됐을 때만 숫자를 낸다

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
        level_db = getattr(fused.approach, "level_db", None)
        calibrated = SPL_CALIBRATED
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
            motion=fused.approach.motion,
            is_horn=is_horn,
            level_db=level_db,
            level_text=_level_text(level_db, calibrated),
            spl_calibrated=calibrated,
        )
