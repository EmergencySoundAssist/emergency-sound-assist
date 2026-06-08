"""
백본(특징 추출기). 입력 → 특징 벡터.

- ScratchCNN: 사전학습 없음. mel (B,1,64,32) → (B, 64)
- MobileNetBackbone: 이미지 사전학습. mel을 3채널로 복제 → (B, 576)
  (freeze=True면 백본 동결, False면 전체 finetune)

YAMNet은 TF라 여기 없음 → yamnet.py(임베딩 추출) + head(build.py) 조합으로 처리.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class ScratchCNN(nn.Module):
    """mel-spectrogram (B,1,64,32) → 특징 벡터 (B, 64). 밑바닥 학습."""

    out_features = 64

    def __init__(self):
        super().__init__()

        def block(i: int, o: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(i, o, kernel_size=3, padding=1),
                nn.BatchNorm2d(o),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(1, 16),     # (B,16,32,16)
            block(16, 32),    # (B,32,16,8)
            block(32, 64),    # (B,64,8,4)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)   # (B,64,1,1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)           # (B,64)
        return x


class MobileNetBackbone(nn.Module):
    """MobileNetV3-small(이미지 사전학습) → 특징 벡터 (B, 576)."""

    out_features = 576

    def __init__(self, freeze: bool = False):
        super().__init__()
        m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.features = m.features            # conv 백본 (분류기 제외)
        self.pool = nn.AdaptiveAvgPool2d(1)
        if freeze:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:                   # 1채널 mel → 3채널 복제 (RGB 자리)
            x = x.repeat(1, 3, 1, 1)
        x = self.features(x)
        x = self.pool(x).flatten(1)           # (B,576)
        return x
