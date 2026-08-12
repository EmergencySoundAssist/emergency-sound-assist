import numpy as np

from approach.detector import ApproachDetector
from classifier.inference import BUF_SAMPLES, SpeedEvidence, _OnnxClassifier
from core.types import (
    ApproachResult,
    AudioChunk,
    Motion,
    SAMPLE_RATE,
    SirenSubtype,
    SoundClass,
)
from pipeline.motion_fusion import (
    ConditionalMotionFusion,
    conditional_decision,
)


def _model(direction: int, confidence: float = 0.9, speed: float = 10.0) -> SpeedEvidence:
    rest = (1.0 - confidence) / 2.0
    probs = [rest, rest, rest]
    probs[direction] = confidence
    return SpeedEvidence(speed, tuple(probs))


def _acoustic(
    motion: Motion,
    energy: float = 0.0,
    doppler_confidence: float = 0.0,
    doppler_motion: Motion = Motion.UNKNOWN,
) -> ApproachResult:
    return ApproachResult(
        motion,
        energy_slope=energy,
        doppler_confidence=doppler_confidence,
        doppler_motion=doppler_motion,
    )


def test_model_waits_for_a_real_five_second_window():
    classifier = _OnnxClassifier()
    classifier._last_x = np.zeros((1, 1, 64, 216), dtype=np.float32)
    classifier._buf = np.zeros(BUF_SAMPLES - 1, dtype=np.float32)
    assert not classifier.speed_ready()
    assert classifier.speed_evidence() is None

    classifier._buf = np.zeros(BUF_SAMPLES, dtype=np.float32)
    assert classifier.speed_ready()


def test_standalone_classifier_includes_subtype(monkeypatch):
    import classifier.inference as CI
    monkeypatch.setattr(CI, "GATE_ON", False)     # argmax 경로 계약 (게이트 경로는 별도 테스트)
    classifier = _OnnxClassifier()
    classifier._last_x = np.zeros((1, 1, 64, 216), dtype=np.float32)
    monkeypatch.setattr(
        classifier,
        "analyze",
        lambda _: {"label": SoundClass.SIREN, "conf": 0.9},
    )
    monkeypatch.setattr(
        classifier,
        "_infer_subtype",
        lambda _: (SirenSubtype.AMBULANCE, 0.8),
    )

    result = classifier.infer(AudioChunk(np.zeros(SAMPLE_RATE, dtype=np.float32)))

    assert result.subtype is SirenSubtype.AMBULANCE
    assert result.subtype_confidence == 0.8


def test_warmup_falls_back_to_acoustic_motion():
    decision = conditional_decision(None, _acoustic(Motion.APPROACHING, energy=0.3))
    assert decision.motion is Motion.APPROACHING
    assert decision.source == "acoustic_warmup"


def test_agreement_is_selected_directly():
    decision = conditional_decision(
        _model(1),
        _acoustic(Motion.APPROACHING, energy=0.3),
    )
    assert decision.motion is Motion.APPROACHING
    assert decision.source == "agree"


def test_conflict_without_doppler_keeps_acoustic_primary():
    decision = conditional_decision(
        _model(0, speed=2.0),
        _acoustic(Motion.APPROACHING, energy=0.3),
    )
    assert decision.motion is Motion.APPROACHING
    assert decision.source == "acoustic_primary"


def test_model_and_doppler_must_agree_to_override_acoustic():
    decision = conditional_decision(
        _model(1),
        _acoustic(
            Motion.STEADY,
            energy=0.0,
            doppler_confidence=0.8,
            doppler_motion=Motion.APPROACHING,
        ),
    )
    assert decision.motion is Motion.APPROACHING
    assert decision.source == "model+doppler"


def test_doppler_can_break_opposite_motion_tie():
    decision = conditional_decision(
        _model(1),
        _acoustic(
            Motion.RECEDING,
            energy=-0.15,
            doppler_confidence=0.8,
            doppler_motion=Motion.APPROACHING,
        ),
    )
    assert decision.motion is Motion.APPROACHING
    assert decision.source == "model+doppler"


def test_low_confidence_model_cannot_override_even_with_doppler():
    decision = conditional_decision(
        _model(1, confidence=0.5),
        _acoustic(
            Motion.RECEDING,
            energy=-0.3,
            doppler_confidence=0.8,
            doppler_motion=Motion.APPROACHING,
        ),
    )
    assert decision.motion is Motion.RECEDING
    assert decision.source == "acoustic_primary"


def test_unresolved_opposite_motion_conflict_keeps_acoustic():
    decision = conditional_decision(
        _model(1),
        _acoustic(Motion.RECEDING, energy=-0.15),
    )
    assert decision.motion is Motion.RECEDING
    assert decision.source == "acoustic_primary"


def test_smoother_requires_repeated_change_after_first_result():
    fusion = ConditionalMotionFusion(smooth_size=3)
    assert fusion.update(None, _acoustic(Motion.APPROACHING)).motion is Motion.APPROACHING
    assert fusion.update(None, _acoustic(Motion.STEADY)).motion is Motion.APPROACHING
    assert fusion.update(None, _acoustic(Motion.STEADY)).motion is Motion.STEADY


def test_detector_exposes_energy_and_doppler_diagnostics():
    detector = ApproachDetector()
    t = np.arange(4 * SAMPLE_RATE, dtype=np.float64) / SAMPLE_RATE
    amplitude = np.exp(0.12 * t)
    signal = (amplitude * np.sin(2 * np.pi * 700.0 * t)).astype(np.float32)
    result = ApproachResult(Motion.UNKNOWN)
    for start in range(0, len(signal), SAMPLE_RATE):
        result = detector.update(AudioChunk(signal[start : start + SAMPLE_RATE]))

    assert result.motion is Motion.APPROACHING
    assert result.energy_slope is not None and result.energy_slope > 0.1
    assert result.frequency_slope is not None
    assert result.doppler_confidence is not None
    assert result.tone_ratio is not None and result.tone_ratio > 0.8
    assert result.frequency_r2 is not None


def test_detector_can_keep_recent_audio_when_event_starts():
    detector = ApproachDetector()
    t = np.arange(3 * SAMPLE_RATE, dtype=np.float64) / SAMPLE_RATE
    signal = (np.exp(0.12 * t) * np.sin(2 * np.pi * 700.0 * t)).astype(np.float32)
    detector.observe(AudioChunk(signal))
    detector.reset(keep_buffer=True)

    result = detector.current()
    assert result.motion is Motion.APPROACHING
    assert result.energy_slope is not None and result.energy_slope > 0.1
