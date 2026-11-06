"""공개 사이렌 원본 단위 leave-one-source-out로 C 융합 조건을 검증한다."""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, deque
from pathlib import Path

import numpy as np


LABELS = ("steady", "approaching", "receding")


def _raw(row: dict, params: tuple[float, float, float]) -> str:
    energy_deadband, doppler_min, model_conf_min = params
    probs = row["model_probs"]
    model = LABELS[int(np.argmax(probs))]
    model_conf = max(probs)
    if row["acoustic_motion"] == "unknown":  # 톤/관측시간 게이트는 그대로 유지
        acoustic = "unknown"
    elif row["energy_slope"] > energy_deadband:
        acoustic = "approaching"
    elif row["energy_slope"] < -energy_deadband:
        acoustic = "receding"
    else:
        acoustic = "steady"
    doppler_conf = row["doppler_confidence"]
    doppler = "approaching" if row["frequency_slope"] > 0 else "receding"
    if acoustic == "unknown":
        return model if model_conf >= model_conf_min else "unknown"
    if model == acoustic:
        return acoustic
    if model_conf >= model_conf_min and doppler_conf >= doppler_min and doppler == model:
        return model
    return acoustic


def _predict(rows: list[dict], params: tuple[float, float, float]):
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["scenario"], []).append(row)
    output = []
    for scenario_rows in grouped.values():
        buf: deque[str] = deque(maxlen=3)
        previous = "unknown"
        for row in scenario_rows:
            buf.append(_raw(row, params))
            votes = Counter(v for v in buf if v != "unknown")
            if votes:
                motion, count = votes.most_common(1)[0]
                if count >= 2 or previous == "unknown":
                    previous = motion
            output.append((row, previous))
    return output


def _score(predictions) -> tuple[float, dict[str, float]]:
    recall = {
        label: float(np.mean([pred == label for row, pred in predictions if row["truth"] == label]))
        for label in LABELS
    }
    score = 0.25 * recall["steady"] + 0.50 * recall["approaching"] + 0.25 * recall["receding"]
    return score, recall


def tune(path: Path) -> dict:
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    sources = sorted({r["source_id"] for r in rows})
    grid = list(
        itertools.product(
            (0.10, 0.15, 0.20, 0.25, 0.30, 0.40),
            (0.10, 0.20, 0.30, 0.40, 0.50, 0.60),
            (0.55, 0.65, 0.75, 0.80, 0.85, 0.90),
        )
    )
    held_out = []
    choices = []
    folds = []
    for source in sources:
        train = [r for r in rows if r["source_id"] != source]
        test = [r for r in rows if r["source_id"] == source]
        best = max(grid, key=lambda p: _score(_predict(train, p))[0])
        choices.append(best)
        fold_predictions = _predict(test, best)
        held_out.extend(fold_predictions)
        folds.append({"source": source, "params": best, "test": _score(fold_predictions)})
    common, common_count = Counter(choices).most_common(1)[0]
    result = {
        "folds": folds,
        "most_common_params": common,
        "most_common_count": common_count,
        "source_count": len(sources),
        "cross_validated": _score(held_out),
        "all_data_with_common_params": _score(_predict(rows, common)),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, nargs="?", default=Path("data/evaluation_results_10sources.json"))
    args = parser.parse_args()
    tune(args.path)


if __name__ == "__main__":
    main()
