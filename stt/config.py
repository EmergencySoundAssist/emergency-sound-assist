"""
stt 설정 모음 (이 모듈의 '리모컨').

엔진 종류·모델 크기·언어, 무음 게이트(VAD) 임계값, 발화 버퍼 길이, 인식 품질 옵션을
한곳에 모아둔다. 값 바꿀 일 있으면 이 파일만 고치면 된다.

core/types.py 의 공통 상수(SAMPLE_RATE)는 중복 정의하지 않고 가져와 쓴다.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional

from core.types import SAMPLE_RATE


@dataclass
class STTConfig:
    """STT 동작 파라미터."""

    # ----- 엔진 -----
    # 1차 선택: faster-whisper (오프라인·CPU·한국어 지원·Jetson 이식 가능).
    # 엔진은 transcriber 에서 추상화돼 있어 vosk 등으로 교체 가능.
    engine: str = "faster-whisper"
    # 한국어는 'small'로 인식이 많이 틀림(실측). 'medium' 이상 권장.
    # (속도가 급하면 --model base/small. Jetson(GPU)은 large-v3-turbo 권장)
    model_size: str = "medium"
    language: Optional[str] = "ko"  # None 이면 자동 감지
    # device/compute_type 은 "auto" 권장 → 노트북=cpu/int8, Jetson(GPU)=cuda/float16.
    device: str = "auto"
    compute_type: str = "auto"
    beam_size: int = 1              # 1=greedy(빠름). 정확도 우선이면 for_accuracy()/--accuracy 로 ↑
    cpu_threads: int = 0            # 0=ctranslate2 자동. Jetson(Orin 6코어)은 4~6 으로 ↑
    num_workers: int = 1           # 단일 실시간 스트림은 1

    # ----- 무음 게이트(VAD): 조용하면 엔진을 아예 안 돌려 비용 절약 -----
    # 너무 낮추면(0.005) 노이즈까지 Whisper 로 보내 헛인식(환각)을 유발 → 0.02 로 보수적.
    # 환경에 맞춰 --vad 로 조정(조용한 실내면 ↓, 시끄러운 도로면 ↑).
    vad_rms_threshold: float = 0.02

    # ----- 발화 단위 버퍼링 -----
    max_utterance_seconds: float = 8.0   # 한 발화 최대 길이(넘으면 강제 flush)
    silence_release_chunks: int = 1      # 음성 뒤 무음이 이만큼 연속되면 발화 끝으로 보고 flush
    min_utterance_seconds: float = 0.5   # 이보다 짧게 모인 건 잡음으로 보고 버림

    # ----- 인식 품질 옵션 -----
    # normalize_audio: 기본 OFF. 조용하면 증폭하지만, 노이즈까지 키우고 클리핑(왜곡) → 인식 악화.
    # 입력이 일관되게 너무 작을 때만 --normalize 로 켤 것.
    normalize_audio: bool = False
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
        """정확도 우선 프로파일(속도는 양보).

        beam_size 5(greedy→탐색). 기존 설정(base)을 받으면 device/model 등은 유지하고
        beam 만 올린다. (정규화는 역효과라 켜지 않음 — 필요하면 --normalize)
        """
        cfg = dataclasses.replace(base) if base is not None else cls()
        cfg.beam_size = 5
        return cfg

    @classmethod
    def for_jetson(cls, model_size: str = "large-v3-turbo") -> "STTConfig":
        """Jetson Orin Nano(8GB, GPU) 권장 프로파일.

        large-v3-turbo + cuda + float16 — 노이즈 한국어에 최적(풀 인코더+빠른 디코더, ~1.6GB).
        통합 8GB 가 빡빡하면 compute_type="int8_float16" 으로(OOM 폴백이 자동 처리).
        ⚠️ 보드는 오프라인 → 모델 미리 받아 복사. 모델 선택·다운로드 → docs/stt/jetson.md
        """
        return cls(model_size=model_size, device="cuda", compute_type="float16")
