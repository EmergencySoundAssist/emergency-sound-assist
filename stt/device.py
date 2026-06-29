"""
실행 디바이스 자동 결정 (노트북 CPU ↔ Jetson CUDA 를 같은 코드로).

config 의 device/compute_type 을 "auto" 로 두면, faster-whisper 의 백엔드인
CTranslate2 에게 직접 'CUDA 장치가 보이는지' 물어 cuda/cpu 를 고른다.
 - 노트북(ctranslate2 미설치 또는 GPU 없음)  → cpu / int8
 - Jetson(ctranslate2 CUDA 빌드 + GPU)        → cuda / float16

resolve_runtime 은 순수 함수라 테스트가 쉽고, 실제 감지는 cuda_available 이 맡는다.
"""

from __future__ import annotations

from typing import Optional, Tuple


def cuda_available() -> bool:
    """CTranslate2 가 실제로 쓸 수 있는 CUDA 장치가 있는지.

    ctranslate2 가 없거나(노트북) CUDA 빌드가 아니면 조용히 False.
    이 검사는 빌드+장치를 함께 보므로, 'GPU 있는데 CPU 빌드' 같은 경우도 False 가 된다.
    """
    try:
        import ctranslate2  # 지연 import: 없으면 그냥 CPU
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def resolve_runtime(
    device: str,
    compute_type: str,
    cuda: Optional[bool] = None,
) -> Tuple[str, str]:
    """("auto"|"cuda"|"cpu", "auto"|...) → 실제 (device, compute_type).

    cuda 를 명시하면 그 값을 쓰고(테스트용), None 이면 cuda_available() 로 감지.
    compute_type="auto" 는 cuda→float16, cpu→int8 로 정한다.
    """
    if cuda is None:
        cuda = cuda_available()

    dev = ("cuda" if cuda else "cpu") if device == "auto" else device

    if compute_type != "auto":
        ct = compute_type
    elif dev == "cuda":
        # Orin Nano(Ampere/SM8.7) 설정: float16 — Tensor 코어 100% 활용, 최상급 한국어 정확도.
        # 단, 통합 8GB 를 더 쓴다. 여러 모듈을 한 보드에 올려 OOM 이면 "int8_float16" 으로 내릴 것.
        ct = "float16"
    else:
        ct = "int8"      # CPU 는 int8 이 가장 가벼움
    return dev, ct
