"""
세 모듈(분류·방향·접근)을 묶어 FusedResult를 만드는 실시간 파이프라인.

흐름 (docs/architecture.md):
  chunk → ① classifier.infer
        → (긴급음일 때만) ② doa.estimate_direction + ③ ApproachDetector.update
        → FusedResult → 예: "사이렌, 후방, 접근 중"

설계 메모:
  - 접근/방향은 사이렌일 때만 의미 있으므로 is_emergency 게이트로 효율화.
  - ApproachDetector는 상태(추세)를 누적하므로 Pipeline이 인스턴스를 보유.
    긴급음이 끊기면 reset() 으로 이벤트 경계를 끊는다.
"""

from __future__ import annotations

from core.types import (
    AudioChunk,
    FusedResult,
    DirectionResult,
    ApproachResult,
    Direction,
    Motion,
)
from classifier import infer as classify
from doa.estimator import estimate_direction
from approach.detector import ApproachDetector


class Pipeline:
    """청크 단위 실시간 처리기. process(chunk) → FusedResult."""

    def __init__(self) -> None:
        self._approach = ApproachDetector()
        self._active = False

    def process(self, chunk: AudioChunk) -> FusedResult:
        cls = classify(chunk)
        if cls.is_emergency:
            self._active = True
            direction = estimate_direction(chunk)
            approach = self._approach.update(chunk)
        else:
            if self._active:                 # 긴급음 종료 → 추세 상태 초기화
                self._approach.reset()
                self._active = False
            direction = DirectionResult(direction=Direction.UNKNOWN)
            approach = ApproachResult(motion=Motion.UNKNOWN)
        return FusedResult(sound=cls, direction=direction, approach=approach)
