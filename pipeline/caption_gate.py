"""평상시 자막 표시 타이밍 게이트.

- 자막 3초 유지(hold): 새 자막이 나오면 hold_seconds 동안 계속 표시한다.
  (짧으면 읽기 전에 사라진다 — 실차 주행 중 읽을 시간을 주려는 값)
- STT 입력은 끊지 않는다: 유지 중에도 청크를 계속 feed 한다. 자막이 떠 있는 동안
  귀를 닫으면 이어지는 발화가 통째로 사라지기 때문이다. 유지 구간에 완성된 자막은
  대기시켰다가 만료 시 올린다 → 화면은 hold 를 다 채우고, 발화도 잃지 않는다.

재인식 억제는 STT 쪽 VAD(Silero)와 min_confidence 가 담당하므로 입력을 끊을 필요가 없다.

시간은 주입 가능(now)해 단위 테스트가 쉽다. pygame·STT 모델 의존 없음.
"""
from __future__ import annotations

from typing import Optional

from core.types import AudioChunk, SpeechResult


class CaptionGate:
    """평상시 자막의 표시 수명을 관리(입력은 항상 통과)."""

    def __init__(self, hold_seconds: float = 3.0) -> None:
        self._hold = float(hold_seconds)
        self._held: Optional[SpeechResult] = None
        self._pending: Optional[SpeechResult] = None   # 유지 중 도착 → 만료 후 표시
        self._expiry = 0.0

    def update(self, worker, chunk: AudioChunk, now: float) -> Optional[SpeechResult]:
        """이번 청크의 표시용 SpeechResult(없으면 None).

        feed 는 매번 한다. worker.latest() 는 소비형(꺼내면 비워짐)이라 유지 중이라도
        반드시 받아 둬야 자막이 증발하지 않는다 — 받은 건 _pending 에 보관한다.
        """
        worker.feed(chunk)
        new = worker.latest()
        if new is not None and not (new.is_speech and new.text):
            new = None

        if self._held is not None and now < self._expiry:
            if new is not None:
                # ponytail: 유지 중 여러 발화가 오면 최신만 남긴다. 표시 지연이 hold 를
                # 넘지 않는 대신 중간 발화는 버린다. 다 보여줘야 하면 큐로 바꾼다.
                self._pending = new
            return self._held

        promoted = new if new is not None else self._pending
        self._pending = None
        if promoted is not None:
            self._held = promoted
            self._expiry = now + self._hold
            return promoted
        self._held = None
        return None

    def reset(self) -> None:
        """긴급 진입 등 → 유지 자막과 대기 자막을 즉시 버린다."""
        self._held = None
        self._pending = None
        self._expiry = 0.0
