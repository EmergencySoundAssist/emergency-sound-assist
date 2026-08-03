"""
STT 모듈 테스트.

faster-whisper 없이도(오프라인·CPU) 돌아가게, 엔진은 '가짜 엔진'을 주입한다.
검증 대상: SpeechResult 동작 / 무음 게이트 / 발화 버퍼링 / 정규화 / 디바이스·프로파일.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from core.types import AudioChunk, SpeechResult, SAMPLE_RATE
from stt.config import STTConfig
from stt.device import resolve_runtime
from stt.transcriber import (
    Transcriber, transcribe_array, FasterWhisperEngine,
    _normalize_rms, _rms, _fallback_chain, _looks_like_oom,
)
from stt.vad import EnergyVad, SileroVad, WebRtcVad, make_vad
from stt.worker import STTWorker
from audio.capture import iter_chunks_threaded
from pipeline import alert as alert_module
from pipeline.runner import Pipeline


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


def _energy_config(**changes):
    """버퍼 로직 테스트가 로컬 webrtcvad 설치 여부에 흔들리지 않게 한다."""
    cfg = STTConfig(vad_backend="energy")
    for key, value in changes.items():
        setattr(cfg, key, value)
    return cfg


def _wait_until(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.01)
    return predicate()


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
    t = Transcriber(config=_energy_config(), engine=eng)
    r = t.transcribe(_silent_chunk())
    assert r.is_speech is False
    assert eng.calls == 0


# ---------------------------------------------------------------------------
# 발화 버퍼링 + flush
# ---------------------------------------------------------------------------
def test_speech_then_silence_flushes_once():
    eng = FakeEngine(text="앞에 차가 지나갑니다")
    t = Transcriber(config=_energy_config(), engine=eng)

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
    cfg = _energy_config(min_utterance_seconds=0.5)
    eng = FakeEngine()
    t = Transcriber(config=cfg, engine=eng)

    # 0.2초 음성(임계 미만 길이) 후 무음 → 너무 짧아 버림, 엔진 미호출
    t.transcribe(_voice_chunk(seconds=0.2))
    r = t.transcribe(_silent_chunk())
    assert r.is_speech is False
    assert eng.calls == 0


def test_max_utterance_forces_flush():
    cfg = _energy_config(max_utterance_seconds=2.0)
    eng = FakeEngine()
    t = Transcriber(config=cfg, engine=eng)

    t.transcribe(_voice_chunk(seconds=1.0))     # 1초
    r = t.transcribe(_voice_chunk(seconds=1.0))  # 누적 2초 → 강제 flush
    assert eng.calls == 1
    assert r.is_speech is True and r.text != ""


def test_flush_on_stream_end():
    eng = FakeEngine()
    t = Transcriber(config=_energy_config(), engine=eng)
    t.transcribe(_voice_chunk())          # 누적만 (무음이 안 와서 flush 안 됨)
    assert eng.calls == 0
    tail = t.flush()                      # 스트림 종료 → 남은 발화 인식
    assert eng.calls == 1 and tail.is_speech is True


def test_low_confidence_caption_is_suppressed():
    class LowConfidenceEngine(FakeEngine):
        def transcribe(self, samples, sample_rate):
            self.calls += 1
            return "시청해주셔서 감사합니다", 0.2, "ko"

    t = Transcriber(config=_energy_config(min_confidence=0.4), engine=LowConfidenceEngine())
    t.transcribe(_voice_chunk())
    result = t.transcribe(_silent_chunk())
    assert result.is_speech is True
    assert result.text == ""
    assert result.confidence == 0.2


def test_on_status_fires_transcribing_only_when_engine_runs():
    events = []
    eng = FakeEngine()
    t = Transcriber(config=_energy_config(), engine=eng, on_status=events.append)
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
# VAD (교체 가능: webrtcvad / energy)
# ---------------------------------------------------------------------------
class _FakeWebrtc:
    """compute 없이 정해진 값을 돌려주는 가짜 webrtcvad.Vad."""
    def __init__(self, result):
        self._r = result

    def is_speech(self, frame, sr):
        return self._r


def test_energy_vad():
    v = EnergyVad(threshold=0.02)
    assert v.is_speech(np.ones(1000, dtype=np.float32) * 0.2, SAMPLE_RATE) is True
    assert v.is_speech(np.zeros(1000, dtype=np.float32), SAMPLE_RATE) is False


def test_webrtcvad_all_voiced_is_speech():
    v = WebRtcVad(voiced_ratio=0.2, _impl=_FakeWebrtc(True))
    assert v.is_speech(np.ones(SAMPLE_RATE, dtype=np.float32) * 0.2, SAMPLE_RATE) is True


def test_webrtcvad_none_voiced_not_speech():
    v = WebRtcVad(voiced_ratio=0.2, _impl=_FakeWebrtc(False))
    assert v.is_speech(np.ones(SAMPLE_RATE, dtype=np.float32) * 0.2, SAMPLE_RATE) is False


def test_silero_vad_uses_voiced_duration_ratio():
    def timestamps(audio, options, sampling_rate):
        return [{"start": 1000, "end": 5000}]

    samples = np.zeros(SAMPLE_RATE, dtype=np.float32)
    assert SileroVad(voiced_ratio=0.2, _timestamps=timestamps).is_speech(samples, SAMPLE_RATE)
    assert not SileroVad(voiced_ratio=0.3, _timestamps=timestamps).is_speech(samples, SAMPLE_RATE)


def test_make_vad_energy_backend():
    cfg = STTConfig(vad_backend="energy")
    assert isinstance(make_vad(cfg), EnergyVad)


def test_make_vad_rejects_unknown_backend():
    with pytest.raises(ValueError, match="VAD backend"):
        make_vad(STTConfig(vad_backend="unknown"))


def test_make_vad_auto_falls_back_when_no_webrtc(monkeypatch):
    # Silero와 WebRTC import를 모두 실패시켜 auto → energy 폴백 확인
    import builtins
    import stt.vad as vad_module
    real_import = builtins.__import__

    def no_silero(*args, **kwargs):
        raise ImportError("no silero")

    monkeypatch.setattr(vad_module, "SileroVad", no_silero)

    def fake_import(name, *a, **k):
        if name == "webrtcvad":
            raise ImportError("no webrtcvad")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert isinstance(make_vad(STTConfig(vad_backend="auto")), EnergyVad)


# ---------------------------------------------------------------------------
# 스레드 캡처 (변환 중에도 입력 안 끊김)
# ---------------------------------------------------------------------------
def test_iter_chunks_threaded_preserves_order():
    out = list(iter_chunks_threaded(iter(range(50))))
    assert out == list(range(50))


def test_iter_chunks_threaded_propagates_error():
    def bad():
        yield 1
        raise RuntimeError("capture died")

    with pytest.raises(RuntimeError):
        list(iter_chunks_threaded(bad()))


# ---------------------------------------------------------------------------
# 통합 STTWorker: 긴급 reset·예외·큐 밀림
# ---------------------------------------------------------------------------
class _BlockingTranscriber:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.reset_applied = threading.Event()

    def transcribe(self, chunk):
        self.started.set()
        self.release.wait(timeout=2.0)
        return SpeechResult(text="긴급 전 오래된 자막", is_speech=True)

    def reset(self):
        self.reset_applied.set()


def test_worker_reset_discards_inflight_caption():
    transcriber = _BlockingTranscriber()
    worker = STTWorker(transcriber, max_queue=2)
    worker.start()
    try:
        worker.feed(_voice_chunk())
        assert transcriber.started.wait(timeout=1.0)
        worker.reset()                         # 변환 도중 긴급 진입
        transcriber.release.set()
        assert transcriber.reset_applied.wait(timeout=1.0)
        assert worker.latest() is None         # 이전 세대 결과는 게시되지 않음
        assert worker.status()["reset_count"] == 1
    finally:
        transcriber.release.set()
        worker.stop()


class _RecoveringTranscriber:
    def __init__(self):
        self.calls = 0

    def transcribe(self, chunk):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("decoder failed")
        return SpeechResult(text="복구된 자막", is_speech=True)

    def reset(self):
        pass


def test_worker_reports_error_and_recovers_on_next_chunk():
    worker = STTWorker(_RecoveringTranscriber())
    worker.start()
    try:
        worker.feed(_voice_chunk())
        assert _wait_until(lambda: worker.status()["last_error"])
        assert worker.status()["alive"]

        worker.feed(_voice_chunk())
        result = _wait_until(worker.latest)
        assert result is not None and result.text == "복구된 자막"
        assert worker.status()["last_error"] is None
    finally:
        worker.stop()


def test_worker_counts_dropped_chunks_when_queue_is_full():
    transcriber = _BlockingTranscriber()
    worker = STTWorker(transcriber, max_queue=1)
    worker.start()
    try:
        worker.feed(_voice_chunk())
        assert transcriber.started.wait(timeout=1.0)
        worker.feed(_voice_chunk())              # 대기열 1칸 사용
        worker.feed(_voice_chunk())              # 초과 → 드롭 카운트
        assert worker.status()["dropped_chunks"] == 1
    finally:
        transcriber.release.set()
        worker.stop()


class _ResetCountingWorker:
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1

    def feed(self, chunk):
        pass

    def latest(self):
        return None


def test_pipeline_resets_stt_only_on_emergency_transition():
    worker = _ResetCountingWorker()
    pipeline = Pipeline(stt_worker=worker)
    mono = np.zeros(SAMPLE_RATE // 10, dtype=np.float32)

    pipeline._stt_step(mono, SAMPLE_RATE, emergency=True)
    pipeline._stt_step(mono, SAMPLE_RATE, emergency=True)
    assert worker.resets == 1
    pipeline._stt_step(mono, SAMPLE_RATE, emergency=False)
    pipeline._stt_step(mono, SAMPLE_RATE, emergency=True)
    assert worker.resets == 2


# ---------------------------------------------------------------------------
# OOM 자동 폴백 (float16 → int8_float16 → int8 → cpu/int8)
# ---------------------------------------------------------------------------
class _FakeModel:
    def __init__(self, compute_type, text="테스트"):
        self.compute_type = compute_type
        self._text = text

    def transcribe(self, samples, **kwargs):
        self.kwargs = kwargs
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
    assert eng._model.kwargs["vad_filter"] is True


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


# ---------------------------------------------------------------------------
# 대시보드 자막 유지
# ---------------------------------------------------------------------------
def _dashboard_event(kind="none"):
    return alert_module.AlertEvent(
        level="NONE" if kind == "none" else "WARN",
        kind=kind,
        label="" if kind == "none" else "경적",
        margin=0.0,
        onset=False,
        remind=False,
        clear=False,
    )


def test_dashboard_holds_caption_for_reading(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(alert_module.time, "monotonic", lambda: clock[0])
    dashboard = alert_module.DashboardSink(caption_hold_seconds=5.0)

    lines = dashboard._render(
        _dashboard_event(),
        {"speech": SpeechResult(text="앞에 사고가 발생했습니다", is_speech=True)},
    )
    assert any("앞에 사고가 발생했습니다" in line for line in lines)

    clock[0] = 104.9
    lines = dashboard._render(_dashboard_event(), {"speech": None})
    assert any("앞에 사고가 발생했습니다" in line for line in lines)

    clock[0] = 105.0
    lines = dashboard._render(_dashboard_event(), {"speech": None})
    assert any("(음성 없음)" in line for line in lines)


def test_dashboard_new_caption_replaces_and_extends_hold(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(alert_module.time, "monotonic", lambda: clock[0])
    dashboard = alert_module.DashboardSink(caption_hold_seconds=5.0)

    dashboard._render(
        _dashboard_event(),
        {"speech": SpeechResult(text="첫 문장", is_speech=True)},
    )
    clock[0] = 13.0
    dashboard._render(
        _dashboard_event(),
        {"speech": SpeechResult(text="두 번째 문장", is_speech=True)},
    )
    clock[0] = 17.9
    lines = dashboard._render(_dashboard_event(), {"speech": None})

    assert any("두 번째 문장" in line for line in lines)
    assert all("첫 문장" not in line for line in lines)


def test_dashboard_emergency_clears_held_caption(monkeypatch):
    clock = [20.0]
    monkeypatch.setattr(alert_module.time, "monotonic", lambda: clock[0])
    dashboard = alert_module.DashboardSink(caption_hold_seconds=5.0)

    dashboard._render(
        _dashboard_event(),
        {"speech": SpeechResult(text="긴급 전 자막", is_speech=True)},
    )
    dashboard._render(_dashboard_event("horn"), {"speech": None})
    clock[0] = 21.0
    lines = dashboard._render(_dashboard_event(), {"speech": None})

    assert any("(음성 없음)" in line for line in lines)
    assert all("긴급 전 자막" not in line for line in lines)
