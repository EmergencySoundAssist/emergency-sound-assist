"""
모듈(분류·방향·접근·STT)을 묶어 FusedResult를 만드는 실시간 파이프라인.

흐름 (docs/architecture.md):
  chunk → ① classifier.infer (ch0)
        → 긴급(siren/horn): ② 방향 + ③ 접근  ·  STT 멈춤(우선순위 전환)
        → 평상시(noise)   : ④ STT 워커에 청크 전달(자막은 뒤에서)
        → FusedResult → "구급차, 후방, 접근 중"  또는  "… · 자막: …"

분류가 '긴급/평상시 스위치' 역할 (docs 1단계 게이트):
  - 사이렌·경적이면 긴급 경고에 집중하고 STT 는 멈춘다(reset). 긴급↔STT 우선순위 전환.
  - 그 외(noise) 면 STT 워커에 청크를 흘려보낸다. 말소리면 워커가 자막을 만든다.

★ STT 는 **백그라운드 워커(스레드)** 로 돈다 (stt.worker.STTWorker):
  - process() 는 worker.feed()/reset() 만 호출하고 **즉시 반환**한다(블로킹 X).
    → STT 가 인식하는 동안에도 분류·방향·접근은 멈추지 않는다(사이렌 놓침 방지).
  - 완성된 자막은 worker.latest() 로 가져와 FusedResult.speech 에 실어 보낸다.
  - stt_worker=None 이면 STT 없이 동작(기존과 동일).

채널 처리:
  - 다채널 (n, C) 이면 ch0(처리채널)을 분류·접근·STT 에, ch1~4 를 방향(SRP)에 쓴다.
  - 1채널이면 방향은 ReSpeaker 자체 DoA 폴백(없으면 미상).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from core.types import (
    AudioChunk,
    FusedResult,
    DirectionResult,
    ApproachResult,
    SpeechResult,
    Direction,
    Motion,
)
from classifier import infer as classify
from doa.estimator import estimate_direction
from approach.detector import ApproachDetector


class Pipeline:
    """청크 단위 실시간 처리기. process(chunk) → FusedResult.

    stt_worker: stt.worker.STTWorker 인스턴스(선택). None 이면 STT 비활성(자막 없음).
                feed/reset/latest 인터페이스만 쓰므로 비동기 워커가 메인을 막지 않는다.
    """

    def __init__(self, stt_worker: Optional["object"] = None) -> None:
        self._approach = ApproachDetector()
        self._active = False
        self._stt = stt_worker           # 평상시 음성→자막 (백그라운드, 없으면 STT 생략)

    def process(self, chunk: AudioChunk) -> FusedResult:
        mono_chunk = _channel0(chunk)            # 분류·접근·STT 는 ch0(처리채널)만
        cls = classify(mono_chunk)
        speech: Optional[SpeechResult] = None

        if cls.is_emergency:                     # ── 긴급: 경고 집중, STT 멈춤 ──
            self._active = True
            direction = self._direction(chunk)
            approach = self._approach.update(mono_chunk)
            if self._stt is not None:
                self._stt.reset()                # 긴급 진입 → 발화 버퍼 비움(즉시 반환)
        else:                                    # ── 평상시: STT 워커로 자막 ──
            if self._active:                     # 긴급음 종료 → 접근 추세 상태 초기화
                self._approach.reset()
                self._active = False
            direction = DirectionResult(direction=Direction.UNKNOWN)
            approach = ApproachResult(motion=Motion.UNKNOWN)
            if self._stt is not None:
                self._stt.feed(mono_chunk)       # 워커에 전달(블로킹 X)
                speech = self._stt.latest()      # 뒤에서 완성된 자막 있으면 실어 보냄

        return FusedResult(sound=cls, direction=direction, approach=approach, speech=speech)

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
