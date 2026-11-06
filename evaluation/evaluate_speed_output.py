"""차량 속도 단계가 제품 출력에 충분한지 source-wise 교차검증한다."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def _accuracy(pairs: list[tuple[int, int]]) -> float | None:
    return float(np.mean([pred == truth for pred, truth in pairs])) if pairs else None


def _binary_loso(rows: list[dict], feature, positive_tiers: set[int]) -> dict:
    predictions: list[tuple[int, int]] = []
    thresholds: list[float] = []
    for source in sorted({row["source_id"] for row in rows}):
        train = [row for row in rows if row["source_id"] != source]
        test = [row for row in rows if row["source_id"] == source]
        values = sorted({float(feature(row)) for row in train})
        grid = [(a + b) / 2.0 for a, b in zip(values, values[1:])]
        if not grid:
            grid = values or [0.0]

        def train_accuracy(threshold: float) -> float:
            return float(np.mean([
                (feature(row) >= threshold) == (row["truth_speed_tier"] in positive_tiers)
                for row in train
            ]))

        threshold = max(grid, key=train_accuracy)
        thresholds.append(threshold)
        predictions.extend([
            (int(feature(row) >= threshold), int(row["truth_speed_tier"] in positive_tiers))
            for row in test
        ])
    return {
        "accuracy": _accuracy(predictions),
        "threshold_median": float(np.median(thresholds)),
        "threshold_range": [float(min(thresholds)), float(max(thresholds))],
    }


def evaluate(path: Path) -> dict:
    rows = [
        row for row in json.loads(path.read_text(encoding="utf-8"))["rows"]
        if row["truth"] == "approaching"
    ]
    model_pairs = []
    acoustic_pairs = []
    consensus_pairs = []
    for row in rows:
        truth = int(row["truth_speed_tier"])
        model = 1 if row["model_speed"] < 20.0 else 2 if row["model_speed"] < 40.0 else 3
        model_pairs.append((model, truth))
        acoustic = row["acoustic_speed_tier"]
        if acoustic is not None:
            acoustic_pairs.append((int(acoustic), truth))
            if int(acoustic) == model:
                consensus_pairs.append((model, truth))

    tier_counts = Counter(int(row["truth_speed_tier"]) for row in rows)
    majority_baseline = max(tier_counts.values()) / len(rows)
    result = {
        "input": str(path),
        "approaching_ticks": len(rows),
        "three_tier_chance_baseline": majority_baseline,
        "three_tier": {
            "model": {"accuracy": _accuracy(model_pairs), "coverage": 1.0},
            "acoustic": {
                "accuracy": _accuracy(acoustic_pairs),
                "coverage": len(acoustic_pairs) / len(rows),
            },
            "agreement_only": {
                "accuracy": _accuracy(consensus_pairs),
                "coverage": len(consensus_pairs) / len(rows),
            },
        },
        "binary": {},
    }
    features = {
        "model_speed": lambda row: float(row["model_speed"]),
        "energy_slope": lambda row: max(0.0, float(row["energy_slope"])),
        "frequency_slope_abs": lambda row: abs(float(row["frequency_slope"])),
    }
    for label, positive in (("20_vs_40_60", {2, 3}), ("20_40_vs_60", {3})):
        positive_count = sum(row["truth_speed_tier"] in positive for row in rows)
        result["binary"][label] = {
            "majority_baseline": max(positive_count, len(rows) - positive_count) / len(rows),
            "features": {
                name: _binary_loso(rows, feature, positive)
                for name, feature in features.items()
            },
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=Path("data/evaluation_results_10sources.json"),
    )
    args = parser.parse_args()
    evaluate(args.path)


if __name__ == "__main__":
    main()
