"""HudView v2 — angle_deg Optional 필드의 전방호환 매핑 테스트.

feat/hud 현재 상태에서 DirectionResult.angle_deg 가 없을 수 있다. from_fused 가
안전하게 None 을 채우고, 값이 있으면 그대로 흡수함을 검증한다.
"""

from core.types import (
    FusedResult, ClassResult, DirectionResult, ApproachResult, SpeechResult,
    SoundClass, SirenSubtype, Direction, Motion,
)
from hud.viewmodel import HudView


def _fused(angle=None, motion=Motion.APPROACHING, subtype=SirenSubtype.AMBULANCE):
    return FusedResult(
        sound=ClassResult(label=SoundClass.SIREN, confidence=0.9,
                          is_emergency=True, subtype=subtype),
        direction=DirectionResult(direction=Direction.LEFT, angle_deg=angle),
        approach=ApproachResult(motion=motion),
        speech=SpeechResult(is_speech=False),
    )


def test_hudview_has_optional_fields_default_none():
    """angle_deg 는 소스에 없으면 None (현재 feat/hud 상태)."""
    v = HudView.from_fused(_fused())
    assert v.angle_deg is None          # DirectionResult.angle_deg 가 None


def test_hudview_reads_angle_when_present():
    """angle_deg 가 있으면 그대로 전달."""
    v = HudView.from_fused(_fused(angle=-45.0))
    assert v.angle_deg == -45.0
