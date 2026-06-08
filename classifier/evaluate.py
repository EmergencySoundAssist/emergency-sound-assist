"""
평가/비교: 저장된 체크포인트들을 test(fold10)에서 평가해 비교 표를 출력.

학습(train.py)과 별개로, 이미 학습된 모델들을 다시 불러와 공정하게 비교한다.
결과는 콘솔 + outputs/comparison.md 에 저장.

사용법:
    python -m classifier.evaluate
"""

from __future__ import annotations

import sys

try:  # Windows 콘솔(cp949)에서 한글 출력 안전
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from . import config
from . import train
from .models import build as B


def evaluate_all(limit: int | None = None):
    names = [config.IDX_TO_CLASS[i].value for i in range(config.NUM_CLASSES)]
    rows, details = [], []

    for exp in config.EXPERIMENTS:
        ckpt_path = config.CHECKPOINT_DIR / f"{exp.name}.pt"
        if not ckpt_path.exists():
            print(f"[skip] {exp.name}: 체크포인트 없음 (먼저 학습 필요)")
            continue

        loader = train.get_test_loader(exp, config.BATCH_SIZE, limit)
        model = B.build_model(exp).to(config.DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=config.DEVICE)["state_dict"])

        _, acc, f1, (trues, preds) = train.evaluate(model, loader, nn.CrossEntropyLoss())
        prec, rec, f1c, _ = precision_recall_fscore_support(
            trues, preds, labels=list(range(config.NUM_CLASSES)), zero_division=0)
        cm = confusion_matrix(trues, preds, labels=list(range(config.NUM_CLASSES)))

        rows.append((exp.name, acc, f1))
        details.append((exp.name, prec, rec, f1c, cm))

    if not rows:
        print("평가할 체크포인트가 없습니다. 먼저 `python -m classifier.train` 실행.")
        return

    # ---- 콘솔 출력 ----
    rows.sort(key=lambda r: r[2], reverse=True)   # macro-F1 내림차순
    print("\n===== 모델 비교 (test=fold10) =====")
    print(f"{'experiment':28s} {'acc':>6s} {'macroF1':>8s}")
    for name, acc, f1 in rows:
        print(f"{name:28s} {acc:6.3f} {f1:8.3f}")
    print(f"\n>> 추천 메인: {rows[0][0]} (macroF1 {rows[0][2]:.3f})")

    # ---- 마크다운 저장 (보고서용) ----
    config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    md = ["# 모델 비교 결과 (test = fold 10)\n",
          "| experiment | acc | macro-F1 |", "|---|---|---|"]
    for name, acc, f1 in rows:
        md.append(f"| {name} | {acc:.3f} | {f1:.3f} |")
    md.append(f"\n**추천 메인 모델: `{rows[0][0]}` (macro-F1 {rows[0][2]:.3f})**\n")
    md.append("## 클래스별 / 혼동행렬\n")
    for name, prec, rec, f1c, cm in details:
        md.append(f"### {name}")
        md.append("| 클래스 | precision | recall | f1 |")
        md.append("|---|---|---|---|")
        for i, cname in enumerate(names):
            md.append(f"| {cname} | {prec[i]:.3f} | {rec[i]:.3f} | {f1c[i]:.3f} |")
        md.append(f"\n혼동행렬(행=정답,열=예측) `{names}`:\n```")
        for i, cname in enumerate(names):
            md.append(f"{cname:15s} {cm[i].tolist()}")
        md.append("```\n")
    out = config.RESULT_DIR / "comparison.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"\n비교 표 저장: {out}")


if __name__ == "__main__":
    evaluate_all()
