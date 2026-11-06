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

import time
from dataclasses import replace
from typing import Optional

import numpy as np

from core.types import (
    AudioChunk,
    ClassResult,
    FusedResult,
    DirectionResult,
    ApproachResult,
    SpeechResult,
    Direction,
    Motion,
)
from classifier import infer as classify
from classifier.inference import speed_evidence   # 패키지는 infer 만 재노출한다
from doa.estimator import estimate_direction
from approach.detector import ApproachDetector
from pipeline.caption_gate import CaptionGate
from pipeline.motion_fusion import ConditionalMotionFusion


class Pipeline:
    """청크 단위 실시간 처리기. process(chunk) → FusedResult.

    stt_worker: stt.worker.STTWorker 인스턴스(선택). None 이면 STT 비활성(자막 없음).
                feed/reset/latest 인터페이스만 쓰므로 비동기 워커가 메인을 막지 않는다.
    """

    def __init__(
        self,
        stt_worker: Optional["object"] = None,
        hold_seconds: float = 3.0,
        emergency_hangover: float = 2.0,
        clock=None,
    ) -> None:
        self._hangover = float(emergency_hangover)
        self._hold_until = 0.0
        self._last_emergency: Optional[ClassResult] = None
        self._approach = ApproachDetector()
        # 조건부 C 융합: 음량 추세 · 속도방향 모델 · 직접 도플러를 상황별로 골라 쓴다.
        # 창은 약 1.5초 — 청크가 1초이므로 3틱. HUD 가 보는 motion 이 이 결과다.
        self._fusion = ConditionalMotionFusion(smooth_size=3)
        self._active = False
        self._stt = stt_worker           # 평상시 음성→자막 (백그라운드, 없으면 STT 생략)
        self._caption = CaptionGate(hold_seconds)   # 자막 3초 유지 + 표시 중 입력 차단
        self._now = clock if clock is not None else time.monotonic

    def process(self, chunk: AudioChunk) -> FusedResult:
        mono_chunk = _channel0(chunk)            # 분류·접근·STT 는 ch0(처리채널)만
        cls = self._debounce(classify(mono_chunk))
        speech: Optional[SpeechResult] = None

        if cls.is_emergency:                     # ── 긴급: 경고 집중, STT 멈춤 ──
            entering = not self._active          # 긴급 진입 edge 인지
            self._active = True
            direction = self._direction(chunk)
            approach = self._fuse(self._approach.update(mono_chunk))
            # 긴급이 지속되는 동안 매 틱 reset 하면 이미 빈 버퍼를 계속 비우고,
            # 띄워 둔 자막도 반복해서 지운다. 진입하는 순간에만 한 번 비운다.
            if entering and self._stt is not None:
                self._stt.reset()                # 발화 버퍼 비움(즉시 반환)
                self._caption.reset()            # 유지 중이던 자막도 즉시 제거
        else:                                    # ── 평상시: STT 워커로 자막 ──
            if self._active:                     # 긴급음 종료 → 접근 추세 상태 초기화
                self._approach.reset()
                self._fusion.reset()             # 다음 이벤트가 이전 판정을 물려받지 않게
                self._active = False
            direction = DirectionResult(direction=Direction.UNKNOWN)
            approach = ApproachResult(motion=Motion.UNKNOWN)
            if self._stt is not None:
                # 자막 3초 유지 + 표시 중 STT 입력 차단은 CaptionGate 가 담당
                speech = self._caption.update(self._stt, mono_chunk, self._now())

        return FusedResult(sound=cls, direction=direction, approach=approach, speech=speech)

    def _debounce(self, raw: ClassResult) -> ClassResult:
        """분류 깜빡임을 잔향으로 메운다 — 켜기는 즉시, 끄기는 hangover 후.

        사이렌이 이어지는 중에도 분류가 한두 청크 'normal' 로 튀면, 화면은 눈이
        무시하지만 폰 진동(notify.BleSender)은 끊김을 몸이 바로 느낀다. 긴급이 사라진
        뒤 hangover 동안은 마지막 긴급 판정을 그대로 유지한다(차종·신뢰도 포함).

        켜는 쪽은 절대 늦추지 않는다 — 경보 지연은 안전 문제다.

        ponytail: 잔향 타이머만 둔다. pipeline.alert.Gate 가 투표창·리마인더까지 있는
        완성형이지만 로짓 마진과 0.15초 tick 을 전제해서(여기는 확률·1초 청크) 안 맞는다.
        마진을 FusedResult 까지 끌어오게 되면 그때 Gate 로 갈아탄다.
        """
        if raw.is_emergency:
            self._hold_until = self._now() + self._hangover
            self._last_emergency = raw
            return raw
        if self._last_emergency is not None and self._now() < self._hold_until:
            return self._last_emergency
        self._last_emergency = None
        return raw

    def _fuse(self, acoustic: ApproachResult) -> ApproachResult:
        """음량 판단에 속도방향 모델과 직접 도플러를 조건부로 얹어 motion 을 확정한다.

        HUD 는 FusedResult.approach.motion 하나만 읽으므로, 융합 결과를 그 자리에
        돌려놓아야 화면이 실제 판정과 같아진다. proximity·gauge 등 나머지 필드는
        음량 추세에서 그대로 나오므로 건드리지 않는다.

        모델 창(5초)이 덜 찼거나 모델 파일이 없으면 speed_evidence() 가 None 을
        돌려주고, 융합은 음량 판단으로 안전하게 폴백한다.
        """
        decision = self._fusion.update(speed_evidence(), acoustic)
        self._fusion_source = decision.source          # 진단용 — 화면에는 쓰지 않는다
        if decision.motion is acoustic.motion:
            return acoustic
        return replace(acoustic, motion=decision.motion)

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
