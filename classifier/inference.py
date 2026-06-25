"""
① 소리 분류 — 추론 인터페이스.

⚠️ 현재는 PLACEHOLDER 휴리스틱입니다 (학습된 ViT/CNN 모델 미연결).
   교체 경로: ViT-CNN-Attention/train.py 로 가중치 생성 → ONNX → (Jetson) TensorRT 엔진
            → 이 함수 내부를 '엔진 추론'으로 교체. 시그니처(infer: AudioChunk→ClassResult)는 유지.
   교체 시 주의: 모델은 22.05kHz·로그멜(1,64,216) 입력 → 이 함수 입구에서 16k→22.05k 리샘플 +
                멜 전처리 필요. 클래스명 noise → normal_traffic 매핑.

임시 판정: 사이렌은 300~2500Hz에 좁고 강한 '톤'을 가짐 → 대역 내 피크가 중앙값 대비 크게
           솟으면 siren. (실차 환경 오탐 많음 — 파이프라인 연결 검증용 placeholder일 뿐.)
"""

from __future__ import annotations

import numpy as np

from core.types import AudioChunk, ClassResult, SoundClass

_BAND = (300.0, 2500.0)


def _mono(samples: np.ndarray) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float64)
    return x.mean(axis=1) if x.ndim > 1 else x


def infer(chunk: AudioChunk) -> ClassResult:
    """AudioChunk → ClassResult. (현재 placeholder 휴리스틱)"""
    x = _mono(chunk.samples)
    if x.size < 64:
        return ClassResult.from_label(SoundClass.NORMAL_TRAFFIC, 0.0)
    w = np.hanning(x.size)
    spec = np.abs(np.fft.rfft(x * w))
    freqs = np.fft.rfftfreq(x.size, d=1.0 / chunk.sample_rate)
    m = (freqs >= _BAND[0]) & (freqs <= _BAND[1])
    if m.sum() < 3:
        return ClassResult.from_label(SoundClass.NORMAL_TRAFFIC, 0.0)
    b = spec[m]
    prominence = float(b.max() / (np.median(b) + 1e-12))   # 톤성(좁은 피크일수록 큼)
    conf = float(np.clip((prominence - 4.0) / 16.0, 0.0, 1.0))
    if conf >= 0.5:
        return ClassResult.from_label(SoundClass.SIREN, conf)
    return ClassResult.from_label(SoundClass.NORMAL_TRAFFIC, 1.0 - conf)
