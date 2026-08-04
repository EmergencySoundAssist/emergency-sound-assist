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


# ---------------------------------------------------------------------------
# 거리 문구 — 접근 중에도 비어 있지 않아야 한다
# ---------------------------------------------------------------------------
def _approaching(gauge, proximity=None):
    return FusedResult(
        sound=ClassResult.from_label(SoundClass.SIREN, 0.9,
                                     subtype=SirenSubtype.AMBULANCE),
        direction=DirectionResult(direction=Direction.REAR, angle_deg=180.0),
        approach=ApproachResult(motion=Motion.APPROACHING, gauge=gauge,
                                proximity=proximity),
        speech=None,
    )


def test_distance_shows_while_still_approaching():
    """detector 가 최근접 전이라 proximity=None 이어도 게이지로 거리를 말한다."""
    assert HudView.from_fused(_approaching(0.10)).proximity == "원거리"
    assert HudView.from_fused(_approaching(0.50)).proximity == "근거리"
    assert HudView.from_fused(_approaching(0.90)).proximity == "근접"


def test_closest_label_is_reserved_for_a_confirmed_peak():
    """접근 중에는 아무리 가까워도 '최근접'이라 하지 않는다 — 더 가까워질 수 있다."""
    assert HudView.from_fused(_approaching(1.0)).proximity == "근접"
    # 최고점이 확정되면 detector 가 준 라벨을 그대로 쓴다
    assert HudView.from_fused(_approaching(1.0, proximity="최근접")).proximity == "최근접"


def test_detector_label_wins_over_the_gauge_fallback():
    assert HudView.from_fused(_approaching(0.1, proximity="근거리")).proximity == "근거리"


def test_no_distance_when_not_approaching():
    """멀어짐·유지·미상에서는 게이지로 거리를 지어내지 않는다."""
    v = FusedResult(
        sound=ClassResult.from_label(SoundClass.SIREN, 0.9),
        direction=DirectionResult(direction=Direction.REAR),
        approach=ApproachResult(motion=Motion.RECEDING, gauge=0.9),
        speech=None,
    )
    assert HudView.from_fused(v).proximity is None


def test_relative_distance_only_after_the_closest_point():
    """거리비는 최고점이 확정된 뒤에만 값이 있다 — 접근 중에는 None."""
    assert HudView.from_fused(_approaching(0.9)).rel_distance is None
    v = FusedResult(
        sound=ClassResult.from_label(SoundClass.SIREN, 0.9),
        direction=DirectionResult(direction=Direction.REAR),
        approach=ApproachResult(motion=Motion.RECEDING, gauge=0.4,
                                proximity="원거리", rel_distance=3.32),
        speech=None,
    )
    assert HudView.from_fused(v).rel_distance == 3.32


def test_level_text_declares_dbfs_when_uncalibrated():
    """마이크 감도 보정 전에는 단위를 dBFS 로 밝힌다 — dB 로 적으면 물리 음압으로 읽힌다."""
    from hud.viewmodel import _level_text
    from approach.detector import SPL_CALIBRATED
    txt = _level_text(-12.4)
    assert txt is not None
    assert ("dBFS" in txt) is (not SPL_CALIBRATED)


def test_level_text_absent_without_a_reading():
    from hud.viewmodel import _level_text
    assert _level_text(None) is None


def test_level_survives_into_the_view():
    v = FusedResult(
        sound=ClassResult.from_label(SoundClass.SIREN, 0.9),
        direction=DirectionResult(direction=Direction.REAR),
        approach=ApproachResult(motion=Motion.APPROACHING, gauge=0.5, level_db=-12.4),
        speech=None,
    )
    view = HudView.from_fused(v)
    assert view.level_db == -12.4
    assert "12" in view.level_text
