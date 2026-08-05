"""각 방식의 문턱을 동일한 source-wise 절차로 튜닝해 공정 비교한다."""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, deque
from pathlib import Path

import numpy as np

from evaluation.tune_fusion import LABELS, _raw as c_raw


def _model(row: dict) -> str:
    return LABELS[int(np.argmax(row["model_probs"]))]


def _acoustic(row: dict, deadband: float) -> str:
    if row["acoustic_motion"] == "unknown":
        return "unknown"
    slope = row["energy_slope"]
    return "approaching" if slope > deadband else "receding" if slope < -deadband else "steady"


def _predict(rows: list[dict], raw) -> list[tuple[dict, str]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["scenario"], []).append(row)
    output = []
    for scenario_rows in grouped.values():
        buf: deque[str] = deque(maxlen=3)
        previous = "unknown"
        for row in scenario_rows:
            buf.append(raw(row))
            votes = Counter(value for value in buf if value != "unknown")
            if votes:
                motion, count = votes.most_common(1)[0]
                if count >= 2 or previous == "unknown":
                    previous = motion
            output.append((row, previous))
    return output


def _metrics(predictions: list[tuple[dict, str]]) -> dict:
    recall = {
        label: float(np.mean([pred == label for row, pred in predictions if row["truth"] == label]))
        for label in LABELS
    }
    return {
        "safety_score": 0.25 * recall["steady"] + 0.50 * recall["approaching"] + 0.25 * recall["receding"],
        "balanced_accuracy": float(np.mean(list(recall.values()))),
        "recall": recall,
    }


def _cross_validate(rows: list[dict], grid: list[tuple], factory) -> dict:
    held_out: list[tuple[dict, str]] = []
    choices = []
    for source in sorted({row["source_id"] for row in rows}):
        train = [row for row in rows if row["source_id"] != source]
        test = [row for row in rows if row["source_id"] == source]
        best = max(grid, key=lambda params: _metrics(_predict(train, factory(params)))["safety_score"])
        choices.append(best)
        held_out.extend(_predict(test, factory(best)))
    return {
        **_metrics(held_out),
        "most_common_params": list(Counter(choices).most_common(1)[0][0]),
        "same_choice_folds": Counter(choices).most_common(1)[0][1],
    }


def compare(path: Path) -> dict:
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    deadbands = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
    model_confidences = (0.55, 0.65, 0.75, 0.80, 0.85, 0.90)
    doppler_confidences = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60)

    model_predictions = _predict(rows, _model)
    result = {
        "model": _metrics(model_predictions),
        "acoustic": _cross_validate(
            rows,
            [(deadband,) for deadband in deadbands],
            lambda params: lambda row: _acoustic(row, params[0]),
        ),
        "doppler": _cross_validate(
            rows,
            [(confidence,) for confidence in doppler_confidences],
            lambda params: lambda row: (
                ("approaching" if row["frequency_slope"] > 0 else "receding")
                if row["doppler_confidence"] >= params[0]
                else "unknown"
            ),
        ),
        "b_model_primary": _cross_validate(
            rows,
            list(itertools.product(deadbands, model_confidences)),
            lambda params: lambda row: (
                _model(row)
                if max(row["model_probs"]) >= params[1]
                else _acoustic(row, params[0])
            ),
        ),
        "c_conditional": _cross_validate(
            rows,
            list(itertools.product(deadbands, doppler_confidences, model_confidences)),
            lambda params: lambda row: c_raw(row, params),
        ),
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
    compare(parser.parse_args().path)


if __name__ == "__main__":
    main()
