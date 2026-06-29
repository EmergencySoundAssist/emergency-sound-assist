"""
stt 설정 모음 (이 모듈의 '리모컨').

엔진 종류·모델 크기·언어, 무음 게이트(VAD) 임계값, 발화 버퍼 길이,
그리고 운전 상황 '긴급 키워드' 목록을 한곳에 모아둔다.
값 바꿀 일 있으면 이 파일만 고치면 된다.

core/types.py 의 공통 상수(SAMPLE_RATE)는 중복 정의하지 않고 가져와 쓴다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from core.types import SAMPLE_RATE


@dataclass
class STTConfig:
    """STT 동작 파라미터."""

    # ----- 엔진 -----
    # 1차 선택: faster-whisper (오프라인·CPU·한국어 지원·Jetson 이식 가능).
    # 엔진은 transcriber 에서 추상화돼 있어 vosk 등으로 교체 가능.
    engine: str = "faster-whisper"
    model_size: str = "base"        # tiny < base < small ... (클수록 정확·느림)
    language: Optional[str] = "ko"  # None 이면 자동 감지
    # device/compute_type 은 "auto" 권장 → 노트북=cpu/int8, Jetson(GPU)=cuda/float16
    # 으로 stt.device.resolve_runtime 이 알아서 고른다. 강제하려면 "cpu"/"cuda" 지정.
    device: str = "auto"
    compute_type: str = "auto"
    beam_size: int = 1              # 1=greedy(가장 빠름). 정확도 필요하면 ↑
    cpu_threads: int = 0            # 0=ctranslate2 자동. Jetson(Orin 6코어)은 4~6 으로 올리면 CPU 추론 빨라짐
    num_workers: int = 1           # 단일 실시간 스트림은 1. (배치 처리량 늘릴 때만 ↑)

    # ----- 무음 게이트(VAD): 조용하면 엔진을 아예 안 돌려 비용 절약 -----
    vad_rms_threshold: float = 0.01  # float32[-1,1] 기준 RMS. 이 이상이면 '음성'으로 간주

    # ----- 발화 단위 버퍼링 -----
    # 1초 청크는 STT 에 너무 짧다 → 음성이 이어지는 동안 모았다가 한 번에 인식.
    max_utterance_seconds: float = 8.0   # 한 발화 최대 길이(넘으면 강제 flush)
    silence_release_chunks: int = 1      # 음성 뒤 무음이 이만큼 연속되면 발화 끝으로 보고 flush
    min_utterance_seconds: float = 0.5   # 이보다 짧게 모인 건 잡음으로 보고 버림

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    @classmethod
    def for_jetson(cls, model_size: str = "small") -> "STTConfig":
        """Jetson Orin Nano(8GB, GPU) 권장 프로파일.

        GPU 가속 명시(cuda/int8_float16). 한국어는 Whisper 난이도가 높아 "small" 권장
        ("base"는 키워드 검출이 약할 수 있음). 메모리가 빡빡하면 model_size="base" 로,
        최대 정확도가 필요하면 compute_type="float16" 로 올린다.
        (자세한 배포 절차·의존성 충돌 → docs/stt/jetson.md)
        """
        return cls(model_size=model_size, device="cuda", compute_type="int8_float16")


# ---------------------------------------------------------------------------
# 긴급 키워드 (운전 상황에서 청각장애 운전자가 놓치면 위험한 말)
# ---------------------------------------------------------------------------
# 부분 문자열(stem)로 매칭하므로 활용형 일부를 함께 잡는다.
#   예) "비키" 는 "비키세요/비킵니다", "비켜" 는 "비켜요/비켜주세요" 를 커버.
EMERGENCY_KEYWORDS: List[str] = [
    # 긴급 차량
    "구급차", "앰뷸런스", "소방차", "경찰", "순찰차",
    # 비키라는 지시
    "비키", "비켜", "양보", "길 좀",
    # 정지/멈춤 지시
    "정지", "멈춰", "세우세요", "차 세워", "후진",
    # 위험 경고
    "위험", "조심", "사고", "충돌", "급정거",
]
