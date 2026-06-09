"""
YAMNet 임베딩 추출기 (TensorFlow).

tfhub YAMNet은 추론용으로 패키징돼 있어 '전체 finetune'이 까다롭다.
표준이자 실용적인 방식 = **YAMNet을 동결(frozen)** 한 채 임베딩(1024)만 뽑고,
그 위에 torch head를 학습한다. (build.py가 head를 만듦)

- 입력: 16kHz 모노 파형 (n,) float32, 대략 [-1, 1]
- 출력: 1024차원 임베딩 (np.float32) — 시간 프레임 평균

⚠️ 첫 호출 시 인터넷에서 YAMNet 모델을 다운로드한다(약 17MB).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def _ensure_ascii_cache_dir() -> None:
    """TF Hub 캐시 경로를 ASCII(영문) 폴더로 강제.

    Windows 사용자명/경로에 한글 등 비ASCII 문자가 있으면 TF Hub의 파일 IO가
    경로를 처리하지 못해 FailedPreconditionError('... is not a directory')가 난다.
    기본 캐시 경로(임시폴더)에 비ASCII가 섞여 있으면 시스템 드라이브 루트의
    영문 폴더로 우회한다. (사용자가 직접 TFHUB_CACHE_DIR을 지정했다면 존중)
    """
    if os.environ.get("TFHUB_CACHE_DIR"):
        return

    def _is_ascii(s: str) -> bool:
        try:
            s.encode("ascii")
            return True
        except UnicodeEncodeError:
            return False

    default_tmp = os.environ.get("TEMP") or os.environ.get("TMP") or ""
    if default_tmp and _is_ascii(default_tmp):
        cache = Path(default_tmp) / "tfhub_cache"
    else:
        system_drive = os.environ.get("SystemDrive", "C:")
        cache = Path(system_drive + os.sep) / "tfhub_cache"

    cache.mkdir(parents=True, exist_ok=True)
    os.environ["TFHUB_CACHE_DIR"] = str(cache)


_ensure_ascii_cache_dir()

_MODEL = None
_YAMNET_URL = "https://tfhub.dev/google/yamnet/1"


def _load():
    global _MODEL
    if _MODEL is None:
        import tensorflow_hub as hub
        _MODEL = hub.load(_YAMNET_URL)
    return _MODEL


def extract_embedding(waveform_16k: np.ndarray) -> np.ndarray:
    """16kHz 파형 → 1024차원 임베딩 (프레임 평균)."""
    model = _load()
    wav = np.asarray(waveform_16k, dtype=np.float32).reshape(-1)
    # YAMNet 반환: (scores, embeddings, spectrogram). embeddings = (frames, 1024)
    _, embeddings, _ = model(wav)
    return embeddings.numpy().mean(axis=0).astype(np.float32)   # (1024,)
