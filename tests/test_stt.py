"""
STT 모듈 테스트.

faster-whisper 없이도(오프라인·CPU) 돌아가게, 엔진은 '가짜 엔진'을 주입한다.
검증 대상: SpeechResult 동작 / 키워드 스포팅 / 무음 게이트 / 발화 버퍼링.
"""

from __future__ import annotations

import numpy as np

from core.types import AudioChunk, SpeechResult, SAMPLE_RATE
from stt.config import STTConfig, EMERGENCY_PROMPT
from stt.device import resolve_runtime
from stt.keywords import find_keywords
from stt.transcriber import Transcriber, transcribe_array, _normalize_rms, _rms


# ---------------------------------------------------------------------------
# 테스트용 도구
# ---------------------------------------------------------------------------
class FakeEngine:
    """호출 횟수를 세고, 정해진 텍스트를 돌려주는 가짜 STT 엔진."""

    def __init__(self, text="앞에 구급차 지나갑니다", lang="ko"):
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
# SpeechResult
# ---------------------------------------------------------------------------
def test_speechresult_defaults():
    r = SpeechResult()
    assert r.text == "" and r.is_speech is False
    assert r.keywords == [] and r.is_alert is False
    assert r.to_korean() == "(음성 없음)"


def test_speechresult_alert_and_korean():
    r = SpeechResult(text="앞에 구급차 지나갑니다", is_speech=True, keywords=["구급차"])
    assert r.is_alert is True
    assert "구급차" in r.to_korean() and "⚠️긴급" in r.to_korean()


def test_speechresult_speech_without_keyword_not_alert():
    r = SpeechResult(text="날씨가 좋네요", is_speech=True)
    assert r.is_alert is False
    assert r.to_korean() == '"날씨가 좋네요"'


# ---------------------------------------------------------------------------
# 키워드 스포팅
# ---------------------------------------------------------------------------
def test_find_keywords_basic_and_inflection():
    assert find_keywords("앞에 구급차 지나갑니다") == ["구급차"]
    # 변형 "비키" 가 잡히면 대표어 "비키세요" 로 알림
    assert find_keywords("빨리 비키세요!") == ["비키세요"]


def test_find_keywords_dedup_and_order():
    out = find_keywords("정지! 정지! 사고 났어요")
    assert out == ["정지", "사고"]   # 등장 순서, 중복 제거


def test_find_keywords_empty_and_none():
    assert find_keywords("") == []
    assert find_keywords("그냥 평범한 대화") == []


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
def test_speech_then_silence_flushes_once_with_keywords():
    eng = FakeEngine(text="앞에 구급차 지나갑니다")
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
    assert r3.text == "앞에 구급차 지나갑니다"
    assert r3.keywords == ["구급차"] and r3.is_alert is True
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


# ---------------------------------------------------------------------------
# 단발(파일) 인식 헬퍼
# ---------------------------------------------------------------------------
def test_transcribe_array_with_fake_engine():
    eng = FakeEngine(text="빨리 비키세요")
    samples = np.ones(int(SAMPLE_RATE * 1.5), dtype=np.float32) * 0.2
    r = transcribe_array(samples, engine=eng)
    assert r.is_speech is True
    assert r.text == "빨리 비키세요"
    assert r.keywords == ["비키세요"]


# ---------------------------------------------------------------------------
# 디바이스 자동 결정 (노트북 CPU ↔ Jetson CUDA)
# ---------------------------------------------------------------------------
def test_resolve_runtime_auto_cpu_when_no_cuda():
    assert resolve_runtime("auto", "auto", cuda=False) == ("cpu", "int8")


def test_resolve_runtime_auto_cuda_when_present():
    # Orin Nano 8GB 권장: 혼합 int8/FP16
    assert resolve_runtime("auto", "auto", cuda=True) == ("cuda", "int8_float16")


def test_resolve_runtime_explicit_overrides_win():
    # device/compute_type 를 명시하면 감지 결과와 무관하게 그대로 쓴다
    assert resolve_runtime("cpu", "int8", cuda=True) == ("cpu", "int8")
    assert resolve_runtime("cuda", "float16", cuda=False) == ("cuda", "float16")


def test_resolve_runtime_auto_compute_follows_explicit_device():
    # device 만 cuda 로 정하고 compute_type 은 auto → int8_float16
    assert resolve_runtime("cuda", "auto", cuda=False) == ("cuda", "int8_float16")


def test_for_jetson_profile():
    cfg = STTConfig.for_jetson()
    assert cfg.device == "cuda" and cfg.compute_type == "int8_float16"
    assert cfg.model_size == "small"


# ---------------------------------------------------------------------------
# 정확도/범위: fuzzy 키워드 매칭
# ---------------------------------------------------------------------------
def test_fuzzy_keyword_one_char_error():
    # STT 가 1글자 틀려도(구급차→구금차) 알림은 살아야 함
    assert "구급차" in find_keywords("앞에 구금차 지나가요")


def test_fuzzy_keyword_spacing_variants():
    # 띄어쓰기 차이 무시 → 대표어로
    assert "차 세우세요" in find_keywords("차세워 빨리")
    assert "비키세요" in find_keywords("비켜 주세요")


def test_fuzzy_disabled_falls_back_to_substring():
    # fuzzy=False 면 오타는 안 잡힌다(부분 문자열만)
    assert find_keywords("앞에 구금차 지나가요", fuzzy=False) == []
    assert find_keywords("앞에 구급차 지나가요", fuzzy=False) == ["구급차"]


def test_fuzzy_no_false_positive_on_plain_text():
    assert find_keywords("오늘 날씨가 정말 좋네요") == []
    assert find_keywords("그냥 평범한 대화였어요") == []


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
# 정확도/범위 프로파일 + 환각 가드 기본값
# ---------------------------------------------------------------------------
def test_for_accuracy_profile():
    cfg = STTConfig.for_accuracy()
    assert cfg.beam_size == 5
    assert cfg.initial_prompt == EMERGENCY_PROMPT
    assert cfg.normalize_audio is True


def test_for_accuracy_keeps_base_device_model():
    base = STTConfig.for_jetson(model_size="medium")
    acc = STTConfig.for_accuracy(base)
    assert acc.device == "cuda" and acc.model_size == "medium"   # base 유지
    assert acc.beam_size == 5                                    # 정확도만 올림


def test_defaults_are_accuracy_range_leaning():
    cfg = STTConfig()
    assert cfg.model_size == "small"          # base→small (한국어 최소)
    assert cfg.hotwords is not None           # 긴급어 가중 기본 ON
    assert cfg.normalize_audio is True        # 범위↑
    assert cfg.vad_rms_threshold == 0.005     # 낮춰서 먼 음성 도달
    assert cfg.no_speech_threshold == 0.6     # 환각 가드는 유지(끄지 않음)
