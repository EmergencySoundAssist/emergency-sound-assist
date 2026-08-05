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
    assert v.confidence == 0.9


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


def test_horn_sets_is_horn_flag():
    v = HudView.from_fused(_fused(SoundClass.HORN))
    assert v.is_horn is True


def test_non_horn_is_horn_false():
    v = HudView.from_fused(_fused(SoundClass.SIREN, SirenSubtype.AMBULANCE))
    assert v.is_horn is False


def test_normal_with_speech_sets_subtitle():
    sp = SpeechResult(text="비켜주세요", is_speech=True)
    v = HudView.from_fused(_fused(SoundClass.NORMAL_TRAFFIC, speech=sp))
    assert v.emergency is False
    assert v.sound_text == "일반 도로 소음"
    assert v.subtitle == "비켜주세요"


def test_normal_without_speech_blank_subtitle():
    v = HudView.from_fused(_fused(SoundClass.NORMAL_TRAFFIC))
    assert v.subtitle == ""


def test_speech_empty_text_blank_subtitle():
    v = HudView.from_fused(_fused(SoundClass.NORMAL_TRAFFIC,
                                     speech=SpeechResult(text="", is_speech=True)))
    assert v.subtitle == ""


def _fused_with_level(level_db):
    from core.types import (
        ClassResult, SoundClass, DirectionResult, Direction,
        ApproachResult, Motion, FusedResult,
    )
    ap = ApproachResult(motion=Motion.APPROACHING)
    setattr(ap, "level_db", level_db)
    return FusedResult(
        sound=ClassResult.from_label(SoundClass.SIREN, 0.9),
        direction=DirectionResult(direction=Direction.RIGHT),
        approach=ap,
    )


def test_no_number_when_uncalibrated():
    """미보정 dBFS 는 운전자에게 의미 없는 숫자다. 단위를 속이느니 안 낸다."""
    import hud.viewmodel as vm
    view = vm.HudView.from_fused(_fused_with_level(-4.0))
    assert view.spl_calibrated is False
    assert view.level_text is None
    assert view.level_db == -4.0          # 값 자체는 남는다 — 미터가 쓴다


def test_number_appears_once_calibrated(monkeypatch):
    import hud.viewmodel as vm
    monkeypatch.setattr(vm, "SPL_CALIBRATED", True)
    view = vm.HudView.from_fused(_fused_with_level(92.0))
    assert view.spl_calibrated is True
    assert view.level_text == "92"


def test_distance_fields_are_gone():
    """v3 에서 '거리'는 가짜 물리량이라 제거됐다. 되살아나면 실패한다."""
    import hud.viewmodel as vm
    view = vm.HudView.from_fused(_fused_with_level(-4.0))
    for dead in ("gauge", "proximity", "rel_distance", "speed_level"):
        assert not hasattr(view, dead), f"{dead} 가 되살아났다"


def test_level_text_absent_without_a_reading():
    from hud.viewmodel import _level_text
    assert _level_text(None, True) is None


def test_level_survives_into_the_view(monkeypatch):
    import hud.viewmodel as vm
    monkeypatch.setattr(vm, "SPL_CALIBRATED", True)
    v = FusedResult(
        sound=ClassResult.from_label(SoundClass.SIREN, 0.9),
        direction=DirectionResult(direction=Direction.REAR),
        approach=ApproachResult(motion=Motion.APPROACHING, gauge=0.5, level_db=-12.4),
        speech=None,
    )
    view = HudView.from_fused(v)
    assert view.level_db == -12.4
    assert "12" in view.level_text
