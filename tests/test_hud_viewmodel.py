"""HudView.from_fused — enum→표시 매핑. pygame 불필요."""

from core.types import (
    FusedResult, ClassResult, DirectionResult, ApproachResult, SpeechResult,
    SoundClass, SirenSubtype, Direction, Motion,
)
from hud.viewmodel import HudView


def _fused(label, subtype=None, direction=Direction.UNKNOWN,
           motion=Motion.UNKNOWN, speech=None):
    return FusedResult(
        sound=ClassResult.from_label(label, 0.9, subtype),
        direction=DirectionResult(direction=direction),
        approach=ApproachResult(motion=motion),
        speech=speech,
    )


def test_siren_with_subtype_shows_vehicle():
    v = HudView.from_fused(_fused(SoundClass.SIREN, SirenSubtype.AMBULANCE,
                                  Direction.REAR, Motion.APPROACHING))
    assert v.emergency is True
    assert v.sound_text == "구급차"
    assert v.direction is Direction.REAR
    assert v.direction_text == "후방"
    assert v.motion_text == "접근 중"
    assert v.subtitle == ""


def test_siren_without_subtype():
    v = HudView.from_fused(_fused(SoundClass.SIREN))
    assert v.sound_text == "사이렌"


def test_unknown_subtype_is_generic_vehicle():
    v = HudView.from_fused(_fused(SoundClass.SIREN, SirenSubtype.UNKNOWN))
    assert v.sound_text == "긴급차량"


def test_horn_is_emergency():
    v = HudView.from_fused(_fused(SoundClass.HORN))
    assert v.emergency is True
    assert v.sound_text == "경적"


def test_normal_with_speech_sets_subtitle():
    sp = SpeechResult(text="비켜주세요", is_speech=True)
    v = HudView.from_fused(_fused(SoundClass.NORMAL_TRAFFIC, speech=sp))
    assert v.emergency is False
    assert v.sound_text == "일반 도로 소음"
    assert v.subtitle == "비켜주세요"


def test_normal_without_speech_blank_subtitle():
    v = HudView.from_fused(_fused(SoundClass.NORMAL_TRAFFIC))
    assert v.subtitle == ""
