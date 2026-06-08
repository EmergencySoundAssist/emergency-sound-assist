"""
분류 head (백본이 뽑은 특징 벡터 → 3클래스 logits).

- linear: 층 1개 (단순, 데이터 적을 때 안전)
- mlp   : 은닉층 1개 추가 (더 복잡한 경계)
"""

from __future__ import annotations

import torch.nn as nn

from .. import config


def build_head(in_features: int, kind: str = "linear",
               num_classes: int = config.NUM_CLASSES,
               hidden: int = 128, dropout: float = 0.3) -> nn.Module:
    if kind == "linear":
        return nn.Linear(in_features, num_classes)
    if kind == "mlp":
        return nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )
    raise ValueError(f"unknown head kind: {kind}")
