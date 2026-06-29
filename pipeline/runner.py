"""
세 모듈(분류·방향·접근)을 묶어 FusedResult를 만드는 실시간 파이프라인.

흐름 (docs/architecture.md):
  chunk → ① classifier.infer (ch0)
        → (긴급음일 때만) ② 방향 + ③ ApproachDetector.update
        → FusedResult → 예: "구급차, 후방, 접근 중"

채널 처리:
  - chunk.samples 가 다채널 (n, C) 이면 **ch0**(ReSpeaker 빔포밍 처리채널)을 분류·접근에,
    **ch1~4**(원본)를 방향(SRP-PHAT)에 쓴다.  → ReSpeaker 6채널 한 스트림으로 전부.
  - 1채널이면 분류·접근만 동작하고, 방향은 ReSpeaker 자체 DoA 폴백(없으면 미상).

설계 메모:
  - 접근/방향은 사이렌일 때만 의미 있으므로 is_emergency 게이트로 효율화.
  - ApproachDetector는 상태(추세)를 누적하므로 Pipeline이 인스턴스를 보유.
    긴급음이 끊기면 reset() 으로 이벤트 경계를 끊는다.
  - 방향(SRP)은 pyroomacoustics·채널이 없으면 조용히 자체 DoA 폴백 → 절대 안 깨진다.
"""

from __future__ import annotations

import numpy as np

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
        mono_chunk = _channel0(chunk)            # 분류·접근은 ch0(처리채널)만
        cls = classify(mono_chunk)
        if cls.is_emergency:
            self._active = True
            direction = self._direction(chunk)
            approach = self._approach.update(mono_chunk)
        else:
            if self._active:                     # 긴급음 종료 → 추세 상태 초기화
                self._approach.reset()
                self._active = False
            direction = DirectionResult(direction=Direction.UNKNOWN)
            approach = ApproachResult(motion=Motion.UNKNOWN)
        return FusedResult(sound=cls, direction=direction, approach=approach)

    def _direction(self, chunk: AudioChunk) -> DirectionResult:
        """다채널이면 ch1~4 SRP-PHAT, 1채널이면 ReSpeaker 자체 DoA 폴백."""
        raw4 = _raw4(np.asarray(chunk.samples))
        if raw4 is None:
            return estimate_direction(chunk)     # 1채널: USB 자체 DoA (없으면 미상)
        try:
            from doa.multi_source import estimate_multiple_directions
            results = estimate_multiple_directions(raw4, fs=chunk.sample_rate)
        except Exception:
            return estimate_direction(chunk)     # pyroomacoustics 미설치 등 → 폴백
        if not results:
            return DirectionResult(direction=Direction.UNKNOWN)
        angle, direction = results[0]            # 에너지 가장 센 방향
        return DirectionResult(direction=direction, angle_deg=angle)


def _channel0(chunk: AudioChunk) -> AudioChunk:
    """다채널이면 ch0(ReSpeaker 빔포밍 처리채널)만, 1채널이면 그대로."""
    s = np.asarray(chunk.samples)
    mono = s[:, 0] if s.ndim == 2 else s
    return AudioChunk(samples=mono, sample_rate=chunk.sample_rate)


def _raw4(s: np.ndarray):
    """방향용 4채널 추출. ReSpeaker 6채널→ch1~4, 4채널→그대로, 그 외→None."""
    if s.ndim != 2:
        return None
    c = s.shape[1]
    if c >= 5:
        return s[:, 1:5]     # ReSpeaker: ch1~4 원본 마이크
    if c == 4:
        return s             # 4채널 직접 입력(합성/테스트)
    return None
