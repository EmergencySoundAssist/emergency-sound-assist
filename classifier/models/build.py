"""
설정(ExperimentConfig) → 모델 조립.

- scratch / mobilenet: 백본 + head 를 묶은 torch 모델. 입력 = log-mel (B,1,64,32)
- yamnet: head만 반환 (입력 = YAMNet 임베딩 1024). 임베딩은 yamnet.py가 TF로 추출(동결).

각 백본이 요구하는 입력 종류는 input_feature() 로 알려준다.
"""

from __future__ import annotations

import torch.nn as nn

from .. import config
from .backbones import MobileNetBackbone, ScratchCNN
from .heads import build_head

YAMNET_EMBED_DIM = 1024


class ClassifierModel(nn.Module):
    """백본 + head 묶음. forward(log-mel) → logits."""

    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        return self.head(self.backbone(x))


def build_model(exp: config.ExperimentConfig) -> nn.Module:
    """실험 설정 → torch 모델."""
    if exp.backbone == "scratch":
        bb = ScratchCNN()
        return ClassifierModel(bb, build_head(bb.out_features, exp.head))

    if exp.backbone == "mobilenet":
        bb = MobileNetBackbone(freeze=(exp.freeze_mode == "frozen"))
        return ClassifierModel(bb, build_head(bb.out_features, exp.head))

    if exp.backbone == "yamnet":
        # YAMNet(동결) 임베딩 1024 → head 만 학습
        return build_head(YAMNET_EMBED_DIM, exp.head)

    raise ValueError(f"unknown backbone: {exp.backbone}")


def input_feature(exp: config.ExperimentConfig) -> str:
    """이 실험이 필요로 하는 입력 종류."""
    if exp.backbone == "yamnet":
        return "yamnet_embedding"      # yamnet.py로 추출한 1024 벡터
    return "logmel"                    # scratch / mobilenet
