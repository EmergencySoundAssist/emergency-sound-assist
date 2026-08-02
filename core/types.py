"""
공통 데이터 인터페이스 (팀 전체가 공유하는 '약속').

세 기능(분류 / 방향 / 접근)이 각각 독립적으로 개발되더라도,
여기 정의된 형식만 지키면 pipeline 단계에서 문제없이 합쳐진다.

- 오디오 입력 형식: AudioChunk
- 각 독립 모듈의 출력 형식: ClassResult / DirectionResult / ApproachResult / SpeechResult
- 실시간 통합 출력은 pipeline.alert.AlertEvent와 진단 info dict를 사용한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# 공통 오디오 설정 (모든 모듈이 동일하게 사용)
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16_000          # Hz. 음성/환경음 분류에서 일반적으로 충분한 값.
CHUNK_SECONDS = 1.0           # 한 번에 분석하는 오디오 길이(초).


@dataclass
class AudioChunk:
    """마이크에서 들어온 한 조각의 오디오.

    samples: shape (n_samples,) 단일 채널 또는 (n_samples, channels) 다채널.
             분류 모듈은 단일 채널만 쓰면 되고, 방향(doa) 모듈이 다채널을 쓴다.
    """
    samples: np.ndarray
    sample_rate: int = SAMPLE_RATE


# ---------------------------------------------------------------------------
# ① 소리 분류 모듈 출력
# ---------------------------------------------------------------------------
class SoundClass(str, Enum):
    SIREN = "siren"
    HORN = "horn"
    NORMAL_TRAFFIC = "normal_traffic"
    # 이후 데이터가 쌓이면 ambulance / fire_truck / police 로 세분화 확장.


class SirenSubtype(str, Enum):
    """사이렌 차종 — siren 일 때만 채워진다 (ViT subtype_clf 모델 출력 → core.types).

    순서는 ViT subtype_clf.SUBS = [구급차, 경찰차, 소방차] 와 일치.
    경찰↔구급 혼동이 잦아, 확신이 낮으면 UNKNOWN('긴급차량')으로 일반화한다.
    """
    AMBULANCE = "ambulance"     # 구급차
    POLICE = "police"           # 경찰차
    FIRE = "fire"               # 소방차
    UNKNOWN = "unknown"         # 차종 불명(확신 낮음) → '긴급차량'


@dataclass
class ClassResult:
    """분류 모듈(★ 내 담당)의 출력."""
    label: SoundClass
    confidence: float                       # 0.0 ~ 1.0
    is_emergency: bool                       # siren/horn 이면 True
    subtype: Optional[SirenSubtype] = None          # siren 일 때만 차종 (구급/경찰/소방/불명)
    subtype_confidence: Optional[float] = None      # 차종 확률 0~1 (없으면 None)

    @staticmethod
    def from_label(
        label: SoundClass,
        confidence: float,
        subtype: Optional[SirenSubtype] = None,
        subtype_confidence: Optional[float] = None,
    ) -> "ClassResult":
        emergency = label in (SoundClass.SIREN, SoundClass.HORN)
        return ClassResult(
            label=label,
            confidence=confidence,
            is_emergency=emergency,
            subtype=subtype,
            subtype_confidence=subtype_confidence,
        )


# ---------------------------------------------------------------------------
# ② 방향 추정 모듈 출력  (팀원 담당 — 인터페이스만 정의)
# ---------------------------------------------------------------------------
class Direction(str, Enum):
    FRONT = "front"
    REAR = "rear"
    LEFT = "left"
    RIGHT = "right"
    UNKNOWN = "unknown"


@dataclass
class DirectionResult:
    direction: Direction
    angle_deg: Optional[float] = None       # 원시 각도(있으면). 없으면 None.


# ---------------------------------------------------------------------------
# ③ 접근/멀어짐 모듈 출력  (팀원 담당 — 인터페이스만 정의)
# ---------------------------------------------------------------------------
class Motion(str, Enum):
    APPROACHING = "approaching"
    RECEDING = "receding"
    STEADY = "steady"
    UNKNOWN = "unknown"


@dataclass
class ApproachResult:
    motion: Motion
    # 아래 speed_level은 과거 호환용 '음량 변화 강도'다. 공개 데이터 검증에서 실제 차량
    # 속도와의 대응이 확인되지 않아 최종 UI에서는 속도로 표시하지 않는다.
    speed_level: Optional[int] = None
    # 상대 근접도 — 이벤트 내 '가장 컸던 순간(=최근접)' 대비 지금 위치. 절대 거리(m)가 아님.
    proximity: Optional[str] = None      # "최근접" / "근거리" / "원거리" (접근 중·미상이면 None)
    rel_distance: Optional[float] = None # 최근접 대비 거리비 (≥1.0, 1.0=가장 가까웠던 지점)
    gauge: Optional[float] = None        # 연속 근접 게이지 0.0~1.0 — 다가오면 차오르고 멀어지면 빠짐
    # 조건부 융합용 원시 진단값. None은 관측 창/톤이 아직 부족하다는 뜻이다.
    energy_slope: Optional[float] = None       # log-대역파워/초: +커짐, -작아짐
    frequency_slope: Optional[float] = None    # 대표 주파수 Hz/초
    doppler_confidence: Optional[float] = None # 단일 피크 추세의 유효도(0~1)
    doppler_motion: Motion = Motion.UNKNOWN    # 직접 도플러 가설(단독 최종판정 금지)
    tone_ratio: Optional[float] = None          # 분석 프레임 중 톤 검출 비율
    frequency_r2: Optional[float] = None        # 주파수 선형 추세 설명력


# ---------------------------------------------------------------------------
# ④ STT(음성→텍스트) 모듈 출력  (확장 기능 — 담당: 천자민)
# ---------------------------------------------------------------------------
@dataclass
class SpeechResult:
    """STT 모듈의 출력 — 주변 음성을 '텍스트로' 바꾼 결과.

    text      : 인식된 문장. 음성이 없거나 인식 실패면 "".
    is_speech : 음성이 감지됐는지(무음 게이트 통과 여부).
    confidence: 0.0 ~ 1.0 (엔진이 주면, 없으면 0.0).
    lang      : 인식 언어 코드(예: "ko"). 모르면 None.
    """
    text: str = ""
    is_speech: bool = False
    confidence: float = 0.0
    lang: Optional[str] = None

    def to_korean(self) -> str:
        """예: '"앞에 차가 지나갑니다"'  /  음성 없으면 '(음성 없음)'."""
        if not self.is_speech or not self.text:
            return "(음성 없음)"
        return f"\"{self.text}\""
