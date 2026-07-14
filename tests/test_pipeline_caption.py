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


def test_pipeline_holds_caption_and_blocks_feed(monkeypatch):
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
    assert w.feeds == 1                                              # 표시 중 입력 차단

    clock["t"] = 3.0
    pipe.process(_chunk())
    assert w.feeds == 2                                              # 만료 후 feed 재개
