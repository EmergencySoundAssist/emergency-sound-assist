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

import numpy as np

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
