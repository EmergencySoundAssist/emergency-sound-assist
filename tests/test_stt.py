"""
STT 모듈 테스트.

faster-whisper 없이도(오프라인·CPU) 돌아가게, 엔진은 '가짜 엔진'을 주입한다.
검증 대상: SpeechResult 동작 / 무음 게이트 / 발화 버퍼링 / 정규화 / 디바이스·프로파일.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.types import AudioChunk, SpeechResult, SAMPLE_RATE
from stt.config import STTConfig
from stt.device import resolve_runtime
from stt.transcriber import (
    Transcriber, transcribe_array, FasterWhisperEngine,
    _normalize_rms, _rms, _fallback_chain, _looks_like_oom,
)


# ---------------------------------------------------------------------------
# 테스트용 도구
# ---------------------------------------------------------------------------
class FakeEngine:
    """호출 횟수를 세고, 정해진 텍스트를 돌려주는 가짜 STT 엔진."""

    def __init__(self, text="앞에 차가 지나갑니다", lang="ko"):
        self.text = text
        self.lang = lang
        self.calls = 0
        self.last_len = 0

    def transcribe(self, samples, sample_rate):
        self.calls += 1
        self.last_len = len(samples)
        return self.text, 0.9, self.lang


def _voice_chunk(seconds=1.0, sr=SAMPLE_RATE, amp=0.2):
    """RMS 가 VAD 임계값 위로 오는 '음성처럼 시끄러운' 청크."""
    n = int(sr * seconds)
    x = (np.ones(n, dtype=np.float32) * amp)
    return AudioChunk(samples=x, sample_rate=sr)


def _silent_chunk(seconds=1.0, sr=SAMPLE_RATE):
    n = int(sr * seconds)
    return AudioChunk(samples=np.zeros(n, dtype=np.float32), sample_rate=sr)


# ---------------------------------------------------------------------------
# SpeechResult (순수 텍스트)
# ---------------------------------------------------------------------------
def test_speechresult_defaults():
    r = SpeechResult()
    assert r.text == "" and r.is_speech is False
    assert r.to_korean() == "(음성 없음)"


def test_speechresult_to_korean():
    r = SpeechResult(text="날씨가 좋네요", is_speech=True)
    assert r.to_korean() == '"날씨가 좋네요"'


def test_speechresult_empty_text_is_no_voice():
    r = SpeechResult(text="", is_speech=True)
    assert r.to_korean() == "(음성 없음)"


# ---------------------------------------------------------------------------
# 무음 게이트(VAD): 조용하면 엔진을 부르지 않는다
# ---------------------------------------------------------------------------
def test_silence_does_not_call_engine():
    eng = FakeEngine()
    t = Transcriber(engine=eng)
    r = t.transcribe(_silent_chunk())
    assert r.is_speech is False
    assert eng.calls == 0


# ---------------------------------------------------------------------------
# 발화 버퍼링 + flush
# ---------------------------------------------------------------------------
def test_speech_then_silence_flushes_once():
    eng = FakeEngine(text="앞에 차가 지나갑니다")
    t = Transcriber(engine=eng)

    # 음성 2초 누적 → 아직 인식 전(엔진 호출 0)
    r1 = t.transcribe(_voice_chunk())
    r2 = t.transcribe(_voice_chunk())
    assert r1.is_speech is True and r1.text == ""
    assert eng.calls == 0

    # 무음 1청크 → 발화 끝으로 보고 flush(엔진 1회)
    r3 = t.transcribe(_silent_chunk())
    assert eng.calls == 1
    assert r3.is_speech is True
    assert r3.text == "앞에 차가 지나갑니다"
    assert eng.last_len == int(SAMPLE_RATE * 2)   # 2초어치를 통째로 인식


def test_short_utterance_is_dropped():
    cfg = STTConfig()
    cfg.min_utterance_seconds = 0.5
    eng = FakeEngine()
    t = Transcriber(config=cfg, engine=eng)

    # 0.2초 음성(임계 미만 길이) 후 무음 → 너무 짧아 버림, 엔진 미호출
    t.transcribe(_voice_chunk(seconds=0.2))
    r = t.transcribe(_silent_chunk())
    assert r.is_speech is False
    assert eng.calls == 0


def test_max_utterance_forces_flush():
    cfg = STTConfig()
    cfg.max_utterance_seconds = 2.0
    eng = FakeEngine()
    t = Transcriber(config=cfg, engine=eng)

    t.transcribe(_voice_chunk(seconds=1.0))     # 1초
    r = t.transcribe(_voice_chunk(seconds=1.0))  # 누적 2초 → 강제 flush
    assert eng.calls == 1
    assert r.is_speech is True and r.text != ""


def test_flush_on_stream_end():
    eng = FakeEngine()
    t = Transcriber(engine=eng)
    t.transcribe(_voice_chunk())          # 누적만 (무음이 안 와서 flush 안 됨)
    assert eng.calls == 0
    tail = t.flush()                      # 스트림 종료 → 남은 발화 인식
    assert eng.calls == 1 and tail.is_speech is True


def test_on_status_fires_transcribing_only_when_engine_runs():
    events = []
    eng = FakeEngine()
    t = Transcriber(engine=eng, on_status=events.append)
    t.transcribe(_voice_chunk())          # 버퍼링 중 — 아직 변환 전
    assert events == []
    t.transcribe(_silent_chunk())         # 발화 끝 → flush → 엔진 실행
    assert events == ["transcribing"]


# ---------------------------------------------------------------------------
# 단발(파일) 인식 헬퍼
# ---------------------------------------------------------------------------
def test_transcribe_array_with_fake_engine():
    eng = FakeEngine(text="빨리 와 주세요")
    samples = np.ones(int(SAMPLE_RATE * 1.5), dtype=np.float32) * 0.2
    r = transcribe_array(samples, engine=eng)
    assert r.is_speech is True
    assert r.text == "빨리 와 주세요"


# ---------------------------------------------------------------------------
# 범위: RMS 정규화
# ---------------------------------------------------------------------------
def test_normalize_quiet_signal_amplified_to_target():
    quiet = np.ones(1000, dtype=np.float32) * 0.01     # 아주 작은 신호
    out = _normalize_rms(quiet, target_rms=0.1, max_gain=10.0)
    assert abs(_rms(out) - 0.1) < 1e-3                 # 목표 RMS 근처로 증폭


def test_normalize_caps_gain_to_avoid_noise_blowup():
    tiny = np.ones(1000, dtype=np.float32) * 0.001     # 게인 100배 필요하지만
    out = _normalize_rms(tiny, target_rms=0.1, max_gain=10.0)
    assert abs(_rms(out) - 0.01) < 1e-3                # max_gain=10 → 0.001*10=0.01 에서 멈춤


def test_normalize_clips_to_unit_range():
    loud = np.ones(1000, dtype=np.float32) * 0.5
    out = _normalize_rms(loud, target_rms=0.1, max_gain=10.0)
    assert out.max() <= 1.0 and out.min() >= -1.0


# ---------------------------------------------------------------------------
# 디바이스 자동 결정 (노트북 CPU ↔ Jetson CUDA)
# ---------------------------------------------------------------------------
def test_resolve_runtime_auto_cpu_when_no_cuda():
    assert resolve_runtime("auto", "auto", cuda=False) == ("cpu", "int8")


def test_resolve_runtime_auto_cuda_when_present():
    # Orin Nano(Ampere) 설정: float16 (Tensor 코어 100% 활용)
    assert resolve_runtime("auto", "auto", cuda=True) == ("cuda", "float16")


def test_resolve_runtime_explicit_overrides_win():
    assert resolve_runtime("cpu", "int8", cuda=True) == ("cpu", "int8")
    assert resolve_runtime("cuda", "int8_float16", cuda=False) == ("cuda", "int8_float16")


def test_resolve_runtime_auto_compute_follows_explicit_device():
    assert resolve_runtime("cuda", "auto", cuda=False) == ("cuda", "float16")


# ---------------------------------------------------------------------------
# 프로파일 + 기본값
# ---------------------------------------------------------------------------
def test_for_jetson_profile():
    cfg = STTConfig.for_jetson()
    assert cfg.device == "cuda" and cfg.compute_type == "float16"
    assert cfg.model_size == "large-v3-turbo"


def test_for_accuracy_profile():
    cfg = STTConfig.for_accuracy()
    assert cfg.beam_size == 5


def test_for_accuracy_keeps_base_device_model():
    base = STTConfig.for_jetson(model_size="medium")
    acc = STTConfig.for_accuracy(base)
    assert acc.device == "cuda" and acc.model_size == "medium"   # base 유지
    assert acc.beam_size == 5                                    # 정확도만 올림


def test_quality_defaults():
    cfg = STTConfig()
    assert cfg.model_size == "medium"         # small은 한국어 인식 약함(실측)
    assert cfg.normalize_audio is False       # 정규화는 역효과라 기본 OFF
    assert cfg.vad_rms_threshold == 0.02      # 너무 낮추면 노이즈→환각
    assert cfg.no_speech_threshold == 0.6     # 환각 가드는 유지(끄지 않음)


# ---------------------------------------------------------------------------
# OOM 자동 폴백 (float16 → int8_float16 → int8 → cpu/int8)
# ---------------------------------------------------------------------------
class _FakeModel:
    def __init__(self, compute_type, text="테스트"):
        self.compute_type = compute_type
        self._text = text

    def transcribe(self, samples, **kwargs):
        seg = type("S", (), {"text": self._text, "avg_logprob": -0.2})()
        info = type("I", (), {"language": "ko"})()
        return [seg], info


def _factory_oom_on(oom_types):
    """cuda 에서 compute_type 이 oom_types 에 들면 OOM 을 던지는 가짜 WhisperModel 팩토리.
    (cpu 는 항상 성공 — 최후 폴백이 도달함을 검증하기 위해)"""
    def factory(model_size, device, compute_type, cpu_threads=0, num_workers=1):
        if device == "cuda" and compute_type in oom_types:
            raise RuntimeError("CUDA failed with error out of memory")
        return _FakeModel(compute_type)
    return factory


def test_fallback_chain_cuda_float16():
    assert _fallback_chain("cuda", "float16") == [
        ("cuda", "float16"), ("cuda", "int8_float16"), ("cuda", "int8"), ("cpu", "int8")
    ]


def test_fallback_chain_cpu_is_single():
    assert _fallback_chain("cpu", "int8") == [("cpu", "int8")]


def test_looks_like_oom():
    assert _looks_like_oom(RuntimeError("CUDA failed with error out of memory"))
    assert _looks_like_oom(RuntimeError("an illegal memory access was encountered"))
    assert not _looks_like_oom(ValueError("bad argument"))


def test_engine_load_oom_falls_back_one_step():
    cfg = STTConfig(device="cuda", compute_type="float16")
    eng = FasterWhisperEngine(cfg, model_factory=_factory_oom_on({"float16"}))
    assert (eng._device, eng._compute_type) == ("cuda", "int8_float16")
    text, _, lang = eng.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
    assert text == "테스트" and lang == "ko"


def test_engine_load_oom_falls_back_to_cpu_when_all_cuda_fail():
    cfg = STTConfig(device="cuda", compute_type="float16")
    eng = FasterWhisperEngine(
        cfg, model_factory=_factory_oom_on({"float16", "int8_float16", "int8"}))
    assert (eng._device, eng._compute_type) == ("cpu", "int8")


def test_engine_non_oom_error_is_not_swallowed():
    cfg = STTConfig(device="cuda", compute_type="float16")

    def bad_factory(model_size, device, compute_type, cpu_threads=0, num_workers=1):
        raise ValueError("관계없는 버그")

    with pytest.raises(ValueError):
        FasterWhisperEngine(cfg, model_factory=bad_factory)


def test_engine_transcribe_oom_rebuilds_and_retries():
    # 로드는 float16 성공하지만 transcribe 가 OOM → int8_float16 재빌드 후 복구
    class _FlakyModel:
        def __init__(self, compute_type):
            self.compute_type = compute_type

        def transcribe(self, samples, **kwargs):
            if self.compute_type == "float16":
                raise RuntimeError("CUDA failed: an illegal memory access")
            return _FakeModel(self.compute_type, text="복구됨").transcribe(samples)

    def factory(model_size, device, compute_type, cpu_threads=0, num_workers=1):
        return _FlakyModel(compute_type)

    cfg = STTConfig(device="cuda", compute_type="float16")
    eng = FasterWhisperEngine(cfg, model_factory=factory)
    assert eng._compute_type == "float16"      # 로드는 성공
    text, _, _ = eng.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
    assert text == "복구됨"
    assert eng._compute_type == "int8_float16"  # 변환 OOM 후 폴백됨
