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
    TICK_SECONDS,
    AudioChunk,
    ClassResult,
    FusedResult,
    DirectionResult,
    ApproachResult,
    SpeechResult,
    Direction,
    Motion,
    SirenSubtype,
    SoundClass,
)
from classifier import infer as classify
from classifier.inference import SUBTYPE_CONF, speed_evidence   # 패키지는 infer 만 재노출한다
from doa.estimator import estimate_direction
from approach.detector import ApproachDetector
from pipeline.alert import SubtypeVote
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
        emergency_hangover: Optional[float] = None,
        clock=None,
    ) -> None:
        # 게이트가 켜져 있으면 잔향은 **게이트가 이미 한다**(t_hang=2.5초). 여기서 또
        # 걸면 2.5+2.0=4.5초가 되어, 벤치(게이트만 잰 것)와 다른 동작을 배포하게 된다.
        if emergency_hangover is None:
            from classifier.inference import GATE_ON
            emergency_hangover = 0.0 if GATE_ON else 2.0
        self._hangover = float(emergency_hangover)
        self._hold_until = 0.0
        self._last_emergency: Optional[ClassResult] = None
        self._approach = ApproachDetector()
        # 조건부 C 융합: 음량 추세 · 속도방향 모델 · 직접 도플러를 상황별로 골라 쓴다.
        # 창은 약 1.5초 — 청크가 1초이므로 3틱. HUD 가 보는 motion 이 이 결과다.
        self._fusion = ConditionalMotionFusion(smooth_size=3)
        self._active = False
        self.last_raw: Optional[ClassResult] = None   # 직전 틱의 디바운스 전 판정
        self._stt = stt_worker           # 평상시 음성→자막 (백그라운드, 없으면 STT 생략)
        self._caption = CaptionGate(hold_seconds)   # 자막 3초 유지 + 표시 중 입력 차단
        self._now = clock if clock is not None else time.monotonic
        # ── 두 격자 ──
        # 검출은 청크마다(0.25초) — 경보 지연의 하한을 낮춘다.
        # 방향·접근·STT·기록은 TICK_SECONDS(1초)로 묶는다 — 이 모듈들은 1초 창을
        # 전제로 튜닝돼 있고(STT VAD 비율, 접근 추세 창, 융합 창 3틱=1.5초),
        # 수집기 det_flags 도 '1틱=1초'라 하류 도구가 그 규약에 묶여 있다.
        self._acc: list[np.ndarray] = []
        self._acc_n = 0
        # 수집기에 넘길 **원본 다채널**도 따로 모은다. ch0 만 모으면 ReSpeaker ch1(원본
        # 마이크)이 사라져서, '차종이 실차에서 무너지는 게 ch0 의 빔포밍/AEC 때문인가'를
        # 영영 못 가른다 — 실제로 e2616db 이후 44클립의 ch1 이 그렇게 소실됐다.
        self._acc_raw: list[np.ndarray] = []
        self.full_tick = False           # 이번 process() 가 1초 경계였나 (main 이 읽는다)
        self.tick_chunk: Optional[AudioChunk] = None   # 그 1초의 오디오(ch0 다운믹스)
        self.tick_raw_chunk: Optional[AudioChunk] = None  # 그 1초의 **원본 다채널**(수집기용)
        self.tick_raw: Optional[ClassResult] = None    # 그 1초의 raw 판정(긴급 우선)
        self._tick_raw_acc: Optional[ClassResult] = None
        self._direction = DirectionResult(direction=Direction.UNKNOWN)
        self._approach_res = ApproachResult(motion=Motion.UNKNOWN)
        self._sub_vote = SubtypeVote()           # 차종 시간 다수결 (떨림 억제)

    def process(self, chunk: AudioChunk) -> FusedResult:
        mono_chunk = _channel0(chunk)            # 분류·접근·STT 는 ch0(처리채널)만
        sr = chunk.sample_rate
        # 디바운스 전 원판정도 남긴다 — 수집기(collect)는 벽시계 잔향이 섞이지 않은
        # raw 를 써야 오디오 시간축과 일관된다(--wav 재수집은 실시간보다 빠르다).
        self.last_raw = classify(mono_chunk)     # ← 매 청크(0.25초 격자)
        cls = self._vote_subtype(self._debounce(self.last_raw), self.last_raw)
        # 1초 틱의 대표 판정: 그 안에서 **가장 먼저 나온 긴급**을 남긴다. 수집기가
        # det_flags 를 1초 단위로 적으므로, 4조각 중 하나라도 울리면 그 틱은 울린 것이다.
        if self._tick_raw_acc is None or (
                self.last_raw.is_emergency and not self._tick_raw_acc.is_emergency):
            self._tick_raw_acc = self.last_raw
        speech: Optional[SpeechResult] = None

        # 1초가 모였는지 — 무거운 모듈은 이 경계에서만 돈다
        self._acc.append(np.asarray(mono_chunk.samples))
        self._acc_raw.append(np.asarray(chunk.samples))
        self._acc_n += mono_chunk.samples.size
        self.full_tick = self._acc_n >= int(TICK_SECONDS * sr)
        tick_chunk = None
        if self.full_tick:
            tick_chunk = AudioChunk(samples=np.concatenate(self._acc), sample_rate=sr)
            self.tick_raw_chunk = AudioChunk(
                samples=np.concatenate(self._acc_raw, axis=0), sample_rate=sr)
            self._acc, self._acc_n, self._acc_raw = [], 0, []
            self.tick_chunk = tick_chunk
            self.tick_raw = self._tick_raw_acc
            self._tick_raw_acc = None

        if cls.is_emergency:                     # ── 긴급: 경고 집중, STT 멈춤 ──
            entering = not self._active          # 긴급 진입 edge 인지
            self._active = True
            if tick_chunk is not None:
                self._direction = self._estimate_direction(chunk)
                self._approach_res = self._fuse(self._approach.update(tick_chunk))
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
            self._direction = DirectionResult(direction=Direction.UNKNOWN)
            self._approach_res = ApproachResult(motion=Motion.UNKNOWN)
            if self._stt is not None and tick_chunk is not None:
                # STT VAD 는 1초 청크 비율로 튜닝돼 있다(stt/config.py) — 조각으로
                # 주면 게이트가 흔들리므로 반드시 묶은 뒤 넘긴다.
                speech = self._caption.update(self._stt, tick_chunk, self._now())

        return FusedResult(sound=cls, direction=self._direction,
                           approach=self._approach_res, speech=speech)

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

    # 차종을 한 번이라도 이름 붙이려면 필요한 것: 유효표 최소 개수와, 그 안에서의 우세 비율.
    # 낮게 잡으면 화면이 구급차↔경찰차로 떤다(현장 신고: "핸드폰으로 사이렌 틀면 차종이 튄다").
    SUBTYPE_MIN_VOTES = 5
    SUBTYPE_MAJORITY = 0.7
    _SUBTYPE_ORDER = (SirenSubtype.AMBULANCE, SirenSubtype.POLICE, SirenSubtype.FIRE)

    def _vote_subtype(self, cls: ClassResult, raw: ClassResult) -> ClassResult:
        """차종을 **시간 다수결**로 눌러 준다 — 창 하나의 추정을 그대로 그리지 않는다.

        왜 필요한가: 검출이 빨라지면(early3/early4) 사이렌이 아직 멀고 약할 때부터
        창마다 차종을 뽑는다. 그 창들에서 차종 확률은 근거가 거의 없어서 매 초 답이
        바뀌고, 화면에는 구급차→경찰차→소방차로 떠는 것으로 보인다. 지연을 줄인 것이
        차종 떨림을 **만든** 셈이다.

        ⚠ 투표는 떨림만 없앤다. 정확도를 만들지는 못한다 — 실차 채널에서 차종 분류기는
        8건 중 1건(균형정확도 50% = 정보 없음)이라, 표를 모아도 '안정적으로 틀린' 답이
        될 수 있다. 그래서 기준을 높게(유효표 5개·우세 70%) 두고, 못 넘으면 UNKNOWN
        (화면 '긴급차량')으로 둔다. 확신 없는 차종을 이름 붙여 말하지 않는 쪽이 낫다.
        """
        if not cls.is_emergency or cls.label is not SoundClass.SIREN:
            self._sub_vote.reset()               # 이벤트 종료 — 다음 사이렌에 표가 안 넘어가게
            return cls
        # ★ 표는 **raw(실제 관측)** 에서만 받는다. 디바운스는 잔향 동안 같은 결과 객체를
        #   계속 돌려주므로, 그걸 먹이면 창 하나가 수 표가 되어 "창 하나로 차종을 그리지
        #   않는다"는 이 함수의 목적 자체가 무너진다(0.25초 격자면 한 창이 최대 8표).
        if (raw.is_emergency and raw.label is SoundClass.SIREN
                and raw.subtype in self._SUBTYPE_ORDER):
            probs = np.zeros(len(self._SUBTYPE_ORDER), dtype=np.float32)
            probs[self._SUBTYPE_ORDER.index(raw.subtype)] = float(raw.subtype_confidence or 0.0)
            self._sub_vote.add(probs, SUBTYPE_CONF)
        i, c, n = self._sub_vote.winner()
        if i is None or n < self.SUBTYPE_MIN_VOTES or c < self.SUBTYPE_MAJORITY * n:
            return replace(cls, subtype=SirenSubtype.UNKNOWN, subtype_confidence=None)
        return replace(cls, subtype=self._SUBTYPE_ORDER[i],
                       subtype_confidence=self._sub_vote.winner_conf())

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

    def _estimate_direction(self, chunk: AudioChunk) -> DirectionResult:
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
