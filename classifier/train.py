"""
학습 스크립트.

EXPERIMENTS의 각 실험을 학습한다.
- 공통: CrossEntropyLoss(+class weight), AdamW, ReduceLROnPlateau, Early stopping
- scratch/mobilenet: log-mel 입력
- yamnet: YAMNet(동결)으로 임베딩 추출(캐시) 후 head 학습

사용법:
    python -m classifier.train                 # 전체 실험 본학습
    python -m classifier.train --smoke         # 빠른 스모크(2 epoch, 소수 샘플)
    python -m classifier.train --exp scratch_cnn   # 특정 실험만
"""

from __future__ import annotations

import argparse
import random
import sys

try:  # Windows 콘솔(cp949)에서 한글 출력 안전
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

from . import config
from . import dataset as D
from .models import build as B
from .models import yamnet as Y


# ---------------------------------------------------------------------------
# 재현성
# ---------------------------------------------------------------------------
def set_seed(seed: int = config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# DataLoader 준비 (백본별 입력 종류 분기)
# ---------------------------------------------------------------------------
def _logmel_loader(folds, batch_size, limit, shuffle):
    ds = D.UrbanSound8KDataset(folds, "logmel", limit)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=config.NUM_WORKERS)


def _yamnet_dataset(folds, limit, tag: str = "") -> TensorDataset:
    """YAMNet 임베딩 추출(+캐시, fold 조합별로 캐시) → TensorDataset."""
    config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    key = "-".join(map(str, folds))
    cache = config.RESULT_DIR / f"yamnet_emb_folds{key}_limit{limit}.npz"
    if cache.exists():
        d = np.load(cache)
        X, y = d["X"], d["y"]
    else:
        ds = D.UrbanSound8KDataset(folds, feature="waveform", limit=limit)
        X, y = [], []
        for i in range(len(ds)):
            wav, label = ds[i]
            X.append(Y.extract_embedding(wav.numpy()))
            y.append(label)
            if (i + 1) % 200 == 0:
                print(f"  [YAMNet 임베딩 {tag or key}] {i+1}/{len(ds)}")
        X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)
        np.savez(cache, X=X, y=y)
    return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))


def _yamnet_loader(folds, batch_size, limit, shuffle, tag=""):
    return DataLoader(_yamnet_dataset(folds, limit, tag),
                      batch_size=batch_size, shuffle=shuffle)


def get_loaders(exp: config.ExperimentConfig, batch_size: int, limit: int | None):
    """학습용 (train, val) 로더. train=fold1~8, val=fold9 (학습 중 선택용)."""
    if B.input_feature(exp) == "logmel":
        return (_logmel_loader(config.TRAIN_FOLDS, batch_size, limit, True),
                _logmel_loader(config.VAL_FOLDS, batch_size, limit, False))
    return (_yamnet_loader(config.TRAIN_FOLDS, batch_size, limit, True, "train"),
            _yamnet_loader(config.VAL_FOLDS, batch_size, limit, False, "val"))


def get_test_loader(exp: config.ExperimentConfig,
                    batch_size: int = config.BATCH_SIZE, limit: int | None = None):
    """최종 평가용 test(fold10) 로더."""
    if B.input_feature(exp) == "logmel":
        return _logmel_loader(config.TEST_FOLDS, batch_size, limit, False)
    return _yamnet_loader(config.TEST_FOLDS, batch_size, limit, False, "test")


# ---------------------------------------------------------------------------
# class weight
# ---------------------------------------------------------------------------
def _loader_labels(loader: DataLoader) -> list[int]:
    ds = loader.dataset
    if hasattr(ds, "label_indices"):
        return ds.label_indices()
    return ds.tensors[1].tolist()      # TensorDataset (yamnet)


def _class_weights(labels: list[int]) -> torch.Tensor:
    counts = np.bincount(labels, minlength=config.NUM_CLASSES).astype(np.float64)
    w = counts.sum() / (config.NUM_CLASSES * np.maximum(counts, 1.0))
    return torch.tensor(w, dtype=torch.float32)


# ---------------------------------------------------------------------------
# 평가
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, preds, trues = 0.0, [], []
    for xb, yb in loader:
        xb, yb = xb.to(config.DEVICE), yb.to(config.DEVICE)
        out = model(xb)
        total_loss += criterion(out, yb).item() * len(yb)
        preds.extend(out.argmax(1).cpu().tolist())
        trues.extend(yb.cpu().tolist())
    avg_loss = total_loss / len(trues)
    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average="macro", zero_division=0)
    return avg_loss, acc, f1, (trues, preds)


# ---------------------------------------------------------------------------
# 한 실험 학습
# ---------------------------------------------------------------------------
def train_one(exp: config.ExperimentConfig, epochs: int, batch_size: int, limit: int | None):
    set_seed()
    print(f"\n===== {exp.name}  (backbone={exp.backbone}, head={exp.head}, {exp.freeze_mode}) =====")

    train_loader, val_loader = get_loaders(exp, batch_size, limit)
    model = B.build_model(exp).to(config.DEVICE)

    weight = (_class_weights(_loader_labels(train_loader)).to(config.DEVICE)
              if config.USE_CLASS_WEIGHTS else None)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE,
                                  weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config.SCHED_FACTOR, patience=config.SCHED_PATIENCE)

    # --- 학습: val(fold9)로 선택 (early stop / best / LR) ---
    best_f1, best_state, no_improve = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(config.DEVICE), yb.to(config.DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        val_loss, val_acc, val_f1, _ = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)
        print(f"  epoch {epoch:2d} | val_loss {val_loss:.3f} | val_acc {val_acc:.3f} | val_F1 {val_f1:.3f}")

        if val_f1 > best_f1:
            best_f1, no_improve = val_f1, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= config.EARLY_STOP_PATIENCE:
                print(f"  early stop (val macro-F1 {config.EARLY_STOP_PATIENCE}회 정체)")
                break

    # --- 최종: best 모델을 test(fold10)로 딱 한 번 평가 ---
    model.load_state_dict(best_state)
    test_loader = get_test_loader(exp, batch_size, limit)
    _, test_acc, test_f1, (trues, preds) = evaluate(model, test_loader, criterion)

    # 체크포인트 저장 (보고 점수 = test)
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = config.CHECKPOINT_DIR / f"{exp.name}.pt"
    torch.save({"exp": exp.__dict__, "state_dict": best_state,
                "metrics": {"acc": test_acc, "macro_f1": test_f1, "val_macro_f1": best_f1}}, ckpt)

    # 혼동행렬 (test 기준)
    cm = confusion_matrix(trues, preds, labels=list(range(config.NUM_CLASSES)))
    names = [config.IDX_TO_CLASS[i].value for i in range(config.NUM_CLASSES)]
    print(f"  >> best val_F1 {best_f1:.3f} → TEST acc {test_acc:.3f}, macroF1 {test_f1:.3f}  (저장: {ckpt.name})")
    print("  [test] 혼동행렬 (행=정답, 열=예측):", names)
    for i, rowname in enumerate(names):
        print(f"    {rowname:15s}", cm[i].tolist())

    return {"name": exp.name, "acc": test_acc, "macro_f1": test_f1}


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="빠른 검증(소수 샘플, 적은 epoch)")
    ap.add_argument("--exp", type=str, default=None, help="특정 실험 이름만 학습")
    args = ap.parse_args()

    epochs = config.SMOKE_EPOCHS if args.smoke else config.EPOCHS
    limit = config.SMOKE_MAX_SAMPLES if args.smoke else None

    experiments = config.EXPERIMENTS
    if args.exp:
        experiments = [e for e in experiments if e.name == args.exp]
        if not experiments:
            raise SystemExit(f"실험 없음: {args.exp}")

    print(f"디바이스 {config.DEVICE} | epochs {epochs} | limit {limit} | 실험 {len(experiments)}개")
    results = [train_one(e, epochs, config.BATCH_SIZE, limit) for e in experiments]

    print("\n===== 요약 =====")
    print(f"{'experiment':28s} {'acc':>6s} {'macroF1':>8s}")
    for r in results:
        print(f"{r['name']:28s} {r['acc']:6.3f} {r['macro_f1']:8.3f}")


if __name__ == "__main__":
    main()
