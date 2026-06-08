"""
추론: 학습된 모델 로드 → AudioChunk → ClassResult.

pipeline에서 이걸 호출한다. core.types 의 약속(ClassResult)을 그대로 반환.
기본은 체크포인트 중 macro-F1이 가장 높은 모델을 자동 선택.
"""

from __future__ import annotations

import numpy as np
import torch

from core.types import AudioChunk, ClassResult
from . import config
from . import preprocessing as P
from .models import build as B
from .models import yamnet as Y


def _best_experiment_name() -> str:
    """저장된 체크포인트 중 검증 macro-F1 최고인 실험 이름."""
    best, best_f1 = None, -1.0
    for e in config.EXPERIMENTS:
        p = config.CHECKPOINT_DIR / f"{e.name}.pt"
        if p.exists():
            m = torch.load(p, map_location="cpu")["metrics"]
            if m["macro_f1"] > best_f1:
                best, best_f1 = e.name, m["macro_f1"]
    if best is None:
        raise FileNotFoundError("학습된 체크포인트가 없습니다. 먼저 train.py 실행.")
    return best


class Classifier:
    """소리 분류기. classify(chunk) → ClassResult."""

    def __init__(self, exp_name: str | None = None):
        self.exp = next(e for e in config.EXPERIMENTS
                        if e.name == (exp_name or _best_experiment_name()))
        ckpt = torch.load(config.CHECKPOINT_DIR / f"{self.exp.name}.pt",
                          map_location=config.DEVICE)
        self.model = B.build_model(self.exp).to(config.DEVICE)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    @torch.no_grad()
    def classify(self, chunk: AudioChunk) -> ClassResult:
        if B.input_feature(self.exp) == "logmel":
            x = P.chunk_to_logmel(chunk).unsqueeze(0).to(config.DEVICE)   # (1,1,64,32)
        else:  # yamnet_embedding
            wav = P.chunk_to_waveform(chunk).numpy()
            emb = Y.extract_embedding(wav)
            x = torch.from_numpy(np.asarray(emb)).unsqueeze(0).to(config.DEVICE)  # (1,1024)

        probs = torch.softmax(self.model(x), dim=1)[0]
        idx = int(probs.argmax())
        return ClassResult.from_label(config.IDX_TO_CLASS[idx], float(probs[idx]))
