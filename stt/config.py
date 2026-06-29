"""
stt 설정 모음 (이 모듈의 '리모컨').

엔진 종류·모델 크기·언어, 무음 게이트(VAD) 임계값, 발화 버퍼 길이,
정확도/범위 옵션, 그리고 운전 상황 '긴급 키워드' 목록을 한곳에 모아둔다.
값 바꿀 일 있으면 이 파일만 고치면 된다.

core/types.py 의 공통 상수(SAMPLE_RATE)는 중복 정의하지 않고 가져와 쓴다.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Optional

from core.types import SAMPLE_RATE


# ---------------------------------------------------------------------------
# 긴급 키워드 (운전 상황에서 청각장애 운전자가 놓치면 위험한 말)
# ---------------------------------------------------------------------------
# 대표어(canonical) → 변형/유의어 목록. 매칭은 부분 문자열 + 자모 fuzzy(→ keywords.py),
# 어느 변형이 잡혀도 '대표어' 하나로 알림한다. (비키/비켜 처럼 비슷한 말의 중복·교차오인 방지)
EMERGENCY_KEYWORDS: Dict[str, List[str]] = {
    "구급차": ["구급차", "앰뷸런스", "119"],
    "소방차": ["소방차"],
    "경찰차": ["경찰차", "경찰", "순찰차", "패트카"],
    "사이렌": ["사이렌"],
    "비키세요": ["비키", "비켜", "비켜주세요", "길 비켜", "양보", "길 좀"],
    "정지": ["정지", "멈춰", "멈추세요", "스톱"],
    "차 세우세요": ["차 세워", "세우세요", "정차", "갓길"],
    "후진": ["후진", "뒤로"],
    "위험": ["위험", "위험해"],
    "조심": ["조심", "주의"],
    "사고": ["사고", "추돌", "접촉사고", "부딪", "충돌"],
    "급정거": ["급정거", "급브레이크", "급정지"],
}

# Whisper hotwords: 디코더를 긴급어 쪽으로 가중. 활용형까지 포함, 224토큰 한참 아래로 짧게.
HOTWORD_STR = (
    "사이렌 구급차 앰뷸런스 소방차 경찰차 순찰차 비키세요 비켜주세요 양보 "
    "정지 멈추세요 차 세우세요 후진 위험 조심 사고 충돌 급정거"
)

# initial_prompt: 첫(유일) 디코드 창을 한국어 긴급 도메인으로 프라이밍.
# 가치 큰 단어를 뒤에. 너무 길면 무음에서 환각 유발 → 짧게 유지하고 thresholds 와 같이 쓴다.
EMERGENCY_PROMPT = (
    "도로 위 긴급 상황 안내 방송입니다. 구급차, 소방차, 경찰차, 비키세요, 정지, 위험, 사고, 조심."
)


@dataclass
class STTConfig:
    """STT 동작 파라미터."""

    # ----- 엔진 -----
    # 1차 선택: faster-whisper (오프라인·CPU·한국어 지원·Jetson 이식 가능).
    # 엔진은 transcriber 에서 추상화돼 있어 vosk 등으로 교체 가능.
    engine: str = "faster-whisper"
    # 한국어는 Whisper 난이도가 높아 'base'는 키워드 검출이 약함 → 최소 'small' 권장.
    # (속도가 급하면 --model base. Jetson 정확도 우선이면 medium/large-v3-turbo 평가)
    model_size: str = "small"
    language: Optional[str] = "ko"  # None 이면 자동 감지
    # device/compute_type 은 "auto" 권장 → 노트북=cpu/int8, Jetson(GPU)=cuda/int8_float16.
    device: str = "auto"
    compute_type: str = "auto"
    beam_size: int = 1              # 1=greedy(빠름). 정확도 우선이면 for_accuracy()/--accuracy 로 ↑
    cpu_threads: int = 0            # 0=ctranslate2 자동. Jetson(Orin 6코어)은 4~6 으로 ↑
    num_workers: int = 1           # 단일 실시간 스트림은 1

    # ----- 무음 게이트(VAD): 조용하면 엔진을 아예 안 돌려 비용 절약 -----
    # 0.01→0.005 로 낮춰 먼/조용한 음성도 Whisper 까지 도달(범위↑). 노이즈는 thresholds 가 거른다.
    vad_rms_threshold: float = 0.005

    # ----- 발화 단위 버퍼링 -----
    max_utterance_seconds: float = 8.0   # 한 발화 최대 길이(넘으면 강제 flush)
    silence_release_chunks: int = 1      # 음성 뒤 무음이 이만큼 연속되면 발화 끝으로 보고 flush
    min_utterance_seconds: float = 0.5   # 이보다 짧게 모인 건 잡음으로 보고 버림

    # ----- 정확도/범위 옵션 -----
    hotwords: Optional[str] = HOTWORD_STR    # 긴급어 가중(기본 ON, 저비용·고효과). 엔진 미지원이면 자동 무시.
    initial_prompt: Optional[str] = None     # 도메인 프라이밍(opt-in). hotwords 와 둘 다 과하게 켜지 말 것.
    # normalize_audio: 먼/조용한 음성을 Whisper 전에 목표 음량으로 키워 '범위'를 넓힌다.
    normalize_audio: bool = True
    target_rms: float = 0.1                  # 정규화 목표 RMS(-20 dBFS)
    max_gain: float = 10.0                   # 최대 증폭배수(순수 노이즈 폭주 방지 상한)
    # Whisper 환각 가드(기본값을 명시 → VAD 게이트를 넓혀도 안전). 끄지 말 것.
    no_speech_threshold: float = 0.6
    log_prob_threshold: float = -1.0
    compression_ratio_threshold: float = 2.4

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    @classmethod
    def for_accuracy(cls, base: Optional["STTConfig"] = None) -> "STTConfig":
        """정확도·범위 우선 프로파일(속도는 양보).

        beam_size 5(greedy→탐색) + 한국어 긴급 도메인 initial_prompt 추가.
        hotwords·정규화·낮춘 VAD 임계값은 이미 기본 ON. 기존 설정(base)을 받으면
        device/model 등은 유지하고 위 항목만 올린다.
        (더 강하게는 model_size="medium" 이지만 8GB 메모리·지연 주의 → 보드에서 측정)
        """
        cfg = dataclasses.replace(base) if base is not None else cls()
        cfg.beam_size = 5
        cfg.initial_prompt = EMERGENCY_PROMPT
        cfg.normalize_audio = True
        return cfg

    @classmethod
    def for_jetson(cls, model_size: str = "small") -> "STTConfig":
        """Jetson Orin Nano(8GB, GPU) 권장 프로파일.

        GPU 가속 명시(cuda/int8_float16). 정확도 더 필요하면 model_size="medium"
        (8GB 메모리·지연 보드에서 측정 후). 자세한 배포 → docs/stt/jetson.md
        """
        return cls(model_size=model_size, device="cuda", compute_type="int8_float16")
