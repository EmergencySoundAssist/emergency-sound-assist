"""
UrbanSound8K 데이터셋 → 모델 입력.

CSV에서 라벨을 읽어 3클래스로 매핑하고, fold로 나누고,
요청이 오면 wav를 읽어 전처리해서 (입력 텐서, 라벨)을 돌려준다.
= "원본 데이터를 모델에게 떠먹여주는 다리".

feature 모드:
- "logmel"   : scratch CNN / MobileNet 용 → (1, 64, 32)
- "waveform" : YAMNet 용 → (16000,)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

from . import config
from . import preprocessing as P


class UrbanSound8KDataset(Dataset):
    def __init__(self, folds: list[int], feature: str = "logmel", limit: int | None = None):
        """
        folds: 사용할 fold 번호 (학습=1~9, 평가=10)
        feature: "logmel" | "waveform"
        limit: 스모크 테스트용 샘플 수 제한 (None이면 전체)
        """
        assert feature in ("logmel", "waveform")
        self.feature = feature

        df = pd.read_csv(config.METADATA_CSV)
        # ① 우리 3클래스에 해당하는 행만 (LABEL_MAP에 없는 class는 제외)
        df = df[df["class"].isin(config.LABEL_MAP.keys())]
        # ② 지정한 fold만
        df = df[df["fold"].isin(folds)].reset_index(drop=True)
        # ③ (스모크) 일부만 — 클래스 다양하게 무작위 샘플
        if limit is not None and limit < len(df):
            df = df.sample(n=limit, random_state=config.SEED).reset_index(drop=True)

        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = config.AUDIO_DIR / f"fold{row['fold']}" / row["slice_file_name"]

        # wav 로드 (제각각 형식 → 전처리에서 통일)
        samples, sr = sf.read(path, dtype="float32")

        if self.feature == "logmel":
            x = P.waveform_to_logmel(samples, sr)   # (1, 64, 32)
        else:
            x = P.prepare_waveform(samples, sr)     # (16000,)

        # 라벨: class 문자열 → SoundClass → 인덱스(0,1,2)
        sound_class = config.LABEL_MAP[row["class"]]
        label = config.CLASS_TO_IDX[sound_class]
        return x, label

    def label_indices(self) -> list[int]:
        """전체 샘플의 라벨 인덱스 리스트 (class weight 계산용)."""
        return [config.CLASS_TO_IDX[config.LABEL_MAP[c]] for c in self.df["class"]]

    def class_counts(self) -> dict[str, int]:
        """클래스별 개수 (확인용)."""
        from collections import Counter
        counts = Counter(self.label_indices())
        return {config.IDX_TO_CLASS[i].value: counts.get(i, 0) for i in range(config.NUM_CLASSES)}


def compute_class_weights(dataset: UrbanSound8KDataset) -> torch.Tensor:
    """클래스 불균형 보정용 가중치 (역빈도). horn처럼 적은 클래스에 큰 가중치."""
    labels = dataset.label_indices()
    counts = np.bincount(labels, minlength=config.NUM_CLASSES).astype(np.float64)
    weights = counts.sum() / (config.NUM_CLASSES * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


def make_dataloaders(
    feature: str = "logmel",
    batch_size: int = config.BATCH_SIZE,
    limit: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """학습/평가 DataLoader 생성. (train=fold1~9, test=fold10)"""
    train_ds = UrbanSound8KDataset(config.TRAIN_FOLDS, feature, limit)
    test_ds = UrbanSound8KDataset(config.TEST_FOLDS, feature, limit)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=config.NUM_WORKERS,
    )
    return train_loader, test_loader
