"""평상시 자막 표시 타이밍 게이트.

- 자막 3초 유지(hold): 새 자막이 나오면 hold_seconds 동안 계속 표시한다.
- 표시 중 STT 입력 차단: 유지 구간에는 워커에 청크를 feed 하지 않는다(재인식 억제).

시간은 주입 가능(now)해 단위 테스트가 쉽다. pygame·STT 모델 의존 없음.
"""
from __future__ import annotations

from typing import Optional

from core.types import AudioChunk, SpeechResult


class CaptionGate:
    """평상시 자막의 표시 수명과 STT feed 차단을 관리."""

    def __init__(self, hold_seconds: float = 3.0) -> None:
        self._hold = float(hold_seconds)
        self._held: Optional[SpeechResult] = None
        self._expiry = 0.0

    def update(self, worker, chunk: AudioChunk, now: float) -> Optional[SpeechResult]:
        """이번 청크의 표시용 SpeechResult(없으면 None).

        유지 구간(now < 만료)에는 worker.feed 를 건너뛰어(STT 입력 차단) 마지막
        자막을 그대로 반환한다. 그 외에는 feed → latest 로 새 자막을 확인하고,
        새 자막이 있으면 만료시각을 now + hold_seconds 로 갱신한다.
        """
        if self._held is not None and now < self._expiry:
            return self._held
        worker.feed(chunk)
        new = worker.latest()
        if new is not None and new.is_speech and new.text:
            self._held = new
            self._expiry = now + self._hold
            return new
        self._held = None
        return None

    def reset(self) -> None:
        """긴급 진입 등 → 유지 자막 즉시 제거."""
        self._held = None
        self._expiry = 0.0
