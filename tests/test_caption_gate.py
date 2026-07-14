"""CaptionGate — 3초 자막 유지 + 표시 중 STT 입력 차단 (시간 주입, 순수)."""

import numpy as np

from core.types import AudioChunk, SpeechResult
from pipeline.caption_gate import CaptionGate


class FakeWorker:
    def __init__(self, script):
        self._script = list(script)      # latest() 가 순서대로 반환할 값
        self.feeds = 0

    def feed(self, chunk):
        self.feeds += 1

    def latest(self):
        return self._script.pop(0) if self._script else None


def _chunk():
    return AudioChunk(samples=np.zeros(16000, dtype=np.float32))


def test_holds_caption_for_3s_and_blocks_feed():
    cap = SpeechResult(text="비켜주세요", is_speech=True)
    w = FakeWorker([cap])
    g = CaptionGate(hold_seconds=3.0)
    assert g.update(w, _chunk(), now=0.0) is cap     # 새 자막 표시
    assert w.feeds == 1
    assert g.update(w, _chunk(), now=1.0) is cap     # 유지 구간
    assert g.update(w, _chunk(), now=2.9) is cap
    assert w.feeds == 1                              # 표시 중엔 입력 차단


def test_resumes_after_expiry():
    cap = SpeechResult(text="비켜주세요", is_speech=True)
    w = FakeWorker([cap, None])
    g = CaptionGate(hold_seconds=3.0)
    g.update(w, _chunk(), now=0.0)
    assert g.update(w, _chunk(), now=3.0) is None    # 만료 → feed 재개, 새 자막 없음
    assert w.feeds == 2


def test_new_caption_resets_timer():
    c1 = SpeechResult(text="하나", is_speech=True)
    c2 = SpeechResult(text="둘", is_speech=True)
    w = FakeWorker([c1])
    g = CaptionGate(hold_seconds=3.0)
    g.update(w, _chunk(), now=0.0)                    # c1, 만료 3.0
    w._script = [c2]
    assert g.update(w, _chunk(), now=3.0) is c2       # 새 자막 → 만료 6.0 리셋
    assert g.update(w, _chunk(), now=5.9) is c2
    assert w.feeds == 2                               # c1때1 + c2때1


def test_reset_clears_held():
    cap = SpeechResult(text="비켜", is_speech=True)
    w = FakeWorker([cap])
    g = CaptionGate(hold_seconds=3.0)
    g.update(w, _chunk(), now=0.0)
    g.reset()
    w._script = [None]
    assert g.update(w, _chunk(), now=0.5) is None     # reset 후엔 유지 없음 → feed 재개
    assert w.feeds == 2
