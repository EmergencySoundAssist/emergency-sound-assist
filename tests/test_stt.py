"""
STT 모듈 테스트.

faster-whisper 없이도(오프라인·CPU) 돌아가게, 엔진은 '가짜 엔진'을 주입한다.
검증 대상: SpeechResult 동작 / 키워드 스포팅 / 무음 게이트 / 발화 버퍼링.
"""

from __future__ import annotations

import numpy as np

from core.types import AudioChunk, SpeechResult, SAMPLE_RATE
from stt.config import STTConfig
from stt.device import resolve_runtime
from stt.keywords import find_keywords
from stt.transcriber import Transcriber, transcribe_array


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
    # "비키" stem 이 "비키세요" 를 잡는다
    assert find_keywords("빨리 비키세요!") == ["비키"]


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
    assert r.keywords == ["비키"]


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
