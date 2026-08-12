"""Pipeline ↔ CaptionGate 배선 — classify 를 평상시로 몽키패치(ONNX 불필요)."""

import numpy as np

from core.types import AudioChunk, SpeechResult, SoundClass, ClassResult


class FakeWorker:
    def __init__(self, script):
        self._script = list(script)
        self.feeds = 0
        self.resets = 0

    def feed(self, chunk):
        self.feeds += 1

    def latest(self):
        return self._script.pop(0) if self._script else None

    def reset(self):
        self.resets += 1


def _chunk():
    return AudioChunk(samples=np.zeros(16000, dtype=np.float32))


def test_pipeline_holds_caption_without_blocking_feed(monkeypatch):
    from pipeline import runner
    monkeypatch.setattr(
        runner, "classify",
        lambda ch: ClassResult.from_label(SoundClass.NORMAL_TRAFFIC, 0.9),
    )
    clock = {"t": 0.0}
    w = FakeWorker([SpeechResult(text="비켜주세요", is_speech=True)])
    pipe = runner.Pipeline(stt_worker=w, hold_seconds=3.0, clock=lambda: clock["t"])

    r0 = pipe.process(_chunk())
    assert r0.speech is not None and r0.speech.text == "비켜주세요"
    assert w.feeds == 1

    clock["t"] = 1.0
    r1 = pipe.process(_chunk())
    assert r1.speech is not None and r1.speech.text == "비켜주세요"   # 3초 유지
    assert w.feeds == 2                                              # 유지 중에도 계속 먹인다

    clock["t"] = 3.0
    pipe.process(_chunk())
    assert w.feeds == 3


# ── 두 격자: 검출 0.25초 · 무거운 모듈 1초 ───────────────────────────────

def test_detection_runs_every_chunk_but_heavy_modules_only_on_full_tick(monkeypatch):
    """청크를 0.25초로 잘게 받아 검출 격자를 촘촘하게 하되, 방향·접근·STT·기록은
    1초로 묶는다 — 그 모듈들은 1초 창을 전제로 튜닝돼 있고 수집기 det_flags 규약도
    '1틱=1초'다."""
    import numpy as np
    import pipeline.runner as R
    from core.types import AudioChunk, ClassResult, SoundClass, SAMPLE_RATE

    n_cls = []
    monkeypatch.setattr(R, "classify",
                        lambda c: n_cls.append(c.samples.size) or
                        ClassResult.from_label(SoundClass.NORMAL_TRAFFIC, 0.8))
    p = R.Pipeline()
    quarter = np.zeros(SAMPLE_RATE // 4, dtype=np.float32)
    ticks = [p.process(AudioChunk(samples=quarter, sample_rate=SAMPLE_RATE)).sound
             for _ in range(4)]
    assert len(n_cls) == 4                      # 검출은 매 청크
    assert [p.full_tick] == [True]              # 4번째에서만 1초 경계
    assert p.tick_chunk.samples.size == SAMPLE_RATE   # 오디오는 손실 없이 모인다


def test_tick_raw_keeps_the_earliest_emergency_in_the_second(monkeypatch):
    """1초 안 4조각 중 하나라도 긴급이면 그 틱은 긴급으로 기록돼야 한다 —
    수집기가 det_flags 를 1초 단위로 적기 때문."""
    import numpy as np
    import pipeline.runner as R
    from core.types import AudioChunk, ClassResult, SoundClass, SAMPLE_RATE

    seq = [ClassResult.from_label(SoundClass.NORMAL_TRAFFIC, 0.8),
           ClassResult.from_label(SoundClass.SIREN, 0.93),
           ClassResult.from_label(SoundClass.NORMAL_TRAFFIC, 0.7),
           ClassResult.from_label(SoundClass.NORMAL_TRAFFIC, 0.7)]
    it = iter(seq)
    monkeypatch.setattr(R, "classify", lambda c: next(it))
    p = R.Pipeline()
    quarter = np.zeros(SAMPLE_RATE // 4, dtype=np.float32)
    for _ in range(4):
        p.process(AudioChunk(samples=quarter, sample_rate=SAMPLE_RATE))
    assert p.tick_raw.is_emergency and p.tick_raw.confidence == 0.93
