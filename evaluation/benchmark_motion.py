"""속도 모델·음량·직접 도플러·B/C 융합을 같은 시나리오에서 비교한다."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from approach.detector import (
    ApproachDetector,
    _dominant_freq_energy,
    _slope,
)
from classifier import inference as clf
from core.types import ApproachResult, AudioChunk, Motion
from evaluation.simulation import Scenario, build_or_load_scenarios
from pipeline.motion_fusion import conditional_decision


MOTIONS = ("steady", "approaching", "receding")
DIR_TO_MOTION = MOTIONS


def _softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max())
    return e / e.sum()


@dataclass(frozen=True)
class ModelEvidence:
    motion: str
    confidence: float
    probs: tuple[float, float, float]
    speed: float
    speed_tier: int


class NeuralEstimator:
    """speed_neural_dir ONNX를 완전한 5초 창에서만 실행한다."""

    def __init__(self, sample_rate: int, window_seconds: float = 5.0):
        self.sr = int(sample_rate)
        self.window = round(window_seconds * self.sr)
        self.buf = np.zeros(0, dtype=np.float32)
        self.sess = clf._CLF._speed_session()
        self.input_name = clf._CLF._spd_in
        self.output_names = [o.name for o in self.sess.get_outputs()]

    def update(self, samples: np.ndarray) -> ModelEvidence | None:
        self.buf = np.concatenate([self.buf, np.asarray(samples, dtype=np.float32)])[-self.window :]
        if len(self.buf) < self.window:
            return None
        y = clf._resample(self.buf, self.sr, clf.SR_MODEL)
        mel = clf._logmel(y)[:, -clf.N_FRAMES :]
        if mel.shape[1] < clf.N_FRAMES:
            # 정상적인 완전 5초 창에서는 거의 발생하지 않지만 반올림 차이를 방어한다.
            mel = np.pad(
                mel,
                ((0, 0), (0, clf.N_FRAMES - mel.shape[1])),
                constant_values=clf.PAD_VAL,
            )
        x = np.ascontiguousarray(clf._norm_win(mel)[None, None], dtype=np.float32)
        raw = self.sess.run(None, {self.input_name: x})
        out = {name: np.asarray(value) for name, value in zip(self.output_names, raw)}
        speed = max(0.0, float(out["speed"].reshape(-1)[0]))
        probs = _softmax(out["dir"].reshape(-1))
        idx = int(probs.argmax())
        tier = 1 if speed < 20.0 else 2 if speed < 40.0 else 3
        return ModelEvidence(
            DIR_TO_MOTION[idx],
            float(probs[idx]),
            tuple(float(p) for p in probs),
            speed,
            tier,
        )


@dataclass(frozen=True)
class PhysicsEvidence:
    acoustic_motion: str
    acoustic_confidence: float
    acoustic_speed_tier: int | None
    energy_slope: float
    frequency_slope: float
    doppler_motion: str
    doppler_confidence: float
    doppler_valid: bool
    tone_ratio: float
    frequency_r2: float


class PhysicsEstimator:
    """현재 음량 판정과 독립적인 대표주파수 도플러 증거를 동시에 산출한다."""

    def __init__(self, sample_rate: int, window_seconds: float = 3.0):
        self.sr = int(sample_rate)
        self.maxlen = round(window_seconds * self.sr)
        self.frame = round(0.5 * self.sr)
        self.hop = round(0.25 * self.sr)
        self.buf = np.zeros(0, dtype=np.float64)
        self.approach = ApproachDetector(sample_rate=self.sr)

    def update(self, samples: np.ndarray) -> PhysicsEvidence:
        x = np.asarray(samples, dtype=np.float64)
        self.buf = np.concatenate([self.buf, x])[-self.maxlen :]
        ap = self.approach.update(AudioChunk(np.asarray(samples, dtype=np.float32), self.sr))
        if len(self.buf) < self.frame:
            return PhysicsEvidence("unknown", 0.0, None, 0.0, 0.0, "unknown", 0.0, False, 0.0, 0.0)

        ts, fs, es = [], [], []
        for start in range(0, len(self.buf) - self.frame + 1, self.hop):
            f, e = _dominant_freq_energy(self.buf[start : start + self.frame], self.sr, (300, 2500), 2.5)
            ts.append((start + self.frame / 2) / self.sr)
            fs.append(f)
            es.append(e)
        ts = np.asarray(ts)
        fs = np.asarray(fs)
        es = np.asarray(es)
        tone = np.isfinite(fs)
        energy_ok = np.isfinite(es)
        tone_ratio = float(tone.mean()) if tone.size else 0.0
        eslope = _slope(ts[energy_ok], es[energy_ok]) if energy_ok.sum() >= 2 else 0.0
        fslope = _slope(ts[tone], fs[tone]) if tone.sum() >= 2 else 0.0

        frequency_r2 = 0.0
        if tone.sum() >= 4 and np.ptp(ts[tone]) >= 1.5:
            pred = fslope * ts[tone] + float(np.mean(fs[tone]) - fslope * np.mean(ts[tone]))
            ss_res = float(np.sum((fs[tone] - pred) ** 2))
            ss_tot = float(np.sum((fs[tone] - np.mean(fs[tone])) ** 2))
            frequency_r2 = max(0.0, 1.0 - ss_res / (ss_tot + 1e-12))

        acoustic_motion = ap.motion.value
        # 움직임일 때는 deadband를 벗어난 정도, 유지일 때는 deadband 안쪽 여유를 신뢰도로 둔다.
        deadband = self.approach.energy_deadband
        if acoustic_motion in ("approaching", "receding"):
            acoustic_conf = float(np.clip((abs(eslope) - deadband) / 0.60, 0.0, 1.0))
        elif acoustic_motion == "steady":
            acoustic_conf = float(np.clip(1.0 - abs(eslope) / deadband, 0.0, 1.0))
        else:
            acoustic_conf = 0.0
        acoustic_conf *= min(1.0, tone_ratio / 0.6)
        acoustic_tier = None
        if ap.speed_level is not None:
            acoustic_tier = 1 if ap.speed_level <= 2 else 2 if ap.speed_level == 3 else 3

        # 단일 피크의 선형 추세는 사이렌 자체 변조에 취약하므로 R²와 톤 지속성을 모두 요구한다.
        magnitude = float(np.clip((abs(fslope) - 8.0) / 40.0, 0.0, 1.0))
        doppler_conf = magnitude * min(1.0, tone_ratio / 0.7) * frequency_r2
        doppler_valid = bool(doppler_conf >= 0.20)
        if fslope > 0:
            doppler_motion = "approaching"
        elif fslope < 0:
            doppler_motion = "receding"
        else:
            doppler_motion = "unknown"
        return PhysicsEvidence(
            acoustic_motion,
            acoustic_conf,
            acoustic_tier,
            float(eslope),
            float(fslope),
            doppler_motion,
            float(doppler_conf),
            doppler_valid,
            tone_ratio,
            frequency_r2,
        )


@dataclass(frozen=True)
class Decision:
    motion: str
    speed_tier: int | None
    source: str


def decide_model(model: ModelEvidence, _: PhysicsEvidence) -> Decision:
    return Decision(model.motion, model.speed_tier if model.motion == "approaching" else None, "model")


def decide_acoustic(_: ModelEvidence, physics: PhysicsEvidence) -> Decision:
    return Decision(
        physics.acoustic_motion,
        physics.acoustic_speed_tier if physics.acoustic_motion == "approaching" else None,
        "acoustic",
    )


def decide_doppler(_: ModelEvidence, physics: PhysicsEvidence) -> Decision:
    motion = physics.doppler_motion if physics.doppler_valid else "unknown"
    return Decision(motion, None, "doppler")


def decide_b(model: ModelEvidence, physics: PhysicsEvidence, model_threshold: float = 0.65) -> Decision:
    """B: 신뢰도 문턱을 넘은 모델이 항상 우선, 아니면 음량 폴백."""
    if model.confidence >= model_threshold:
        return Decision(model.motion, model.speed_tier if model.motion == "approaching" else None, "model")
    return decide_acoustic(model, physics)


def decide_c(model: ModelEvidence, physics: PhysicsEvidence) -> Decision:
    """제품 코드와 같은 조건부 C 규칙. 속도 단계는 검증 부족으로 기권한다."""
    neural = clf.SpeedEvidence(model.speed, model.probs)
    acoustic = ApproachResult(
        motion=Motion(physics.acoustic_motion),
        speed_level=physics.acoustic_speed_tier,
        energy_slope=physics.energy_slope,
        frequency_slope=physics.frequency_slope,
        doppler_confidence=physics.doppler_confidence,
        doppler_motion=Motion(physics.doppler_motion),
        tone_ratio=physics.tone_ratio,
        frequency_r2=physics.frequency_r2,
    )
    fused = conditional_decision(neural, acoustic)
    return Decision(fused.motion.value, None, fused.source)


class MajoritySmoother:
    def __init__(self, size: int = 3):
        self.buf: deque[Decision] = deque(maxlen=size)
        self.previous = "unknown"

    def update(self, decision: Decision) -> Decision:
        self.buf.append(decision)
        votes = Counter(d.motion for d in self.buf if d.motion != "unknown")
        if not votes:
            return Decision(self.previous, None, "hold")
        motion, count = votes.most_common(1)[0]
        if count < 2 and self.previous != "unknown":
            motion = self.previous
        else:
            self.previous = motion
        candidates = [d for d in self.buf if d.motion == motion and d.speed_tier is not None]
        tier = None
        if motion == "approaching" and candidates:
            values = sorted(d.speed_tier for d in candidates)
            tier = values[len(values) // 2]
        return Decision(motion, tier, "smoothed")


def _scenario_rows(scenario: Scenario, tick_seconds: float = 0.5) -> list[dict]:
    sr = scenario.sample_rate
    chunk_n = round(tick_seconds * sr)
    neural = NeuralEstimator(sr)
    physics = PhysicsEstimator(sr)
    smoothers = {name: MajoritySmoother() for name in ("model", "acoustic", "doppler", "b", "c")}
    rows: list[dict] = []
    for start in range(0, len(scenario.samples) - chunk_n + 1, chunk_n):
        chunk = scenario.samples[start : start + chunk_n]
        time_s = (start + chunk_n) / sr
        me = neural.update(chunk)
        pe = physics.update(chunk)
        if me is None:  # 모델의 실제 5초 워밍업 이후만 공정 비교
            continue
        raw = {
            "model": decide_model(me, pe),
            "acoustic": decide_acoustic(me, pe),
            "doppler": decide_doppler(me, pe),
            "b": decide_b(me, pe),
            "c": decide_c(me, pe),
        }
        decisions = {name: smoothers[name].update(decision) for name, decision in raw.items()}
        rows.append(
            {
                "scenario": scenario.scenario_id,
                "source_id": scenario.source_id,
                "kind": scenario.kind,
                "time": time_s,
                "truth": scenario.truth_at(time_s),
                "truth_speed_tier": scenario.speed_tier,
                "model_confidence": me.confidence,
                "model_probs": list(me.probs),
                "model_speed": me.speed,
                "energy_slope": pe.energy_slope,
                "acoustic_confidence": pe.acoustic_confidence,
                "acoustic_motion": pe.acoustic_motion,
                "acoustic_speed_tier": pe.acoustic_speed_tier,
                "frequency_slope": pe.frequency_slope,
                "doppler_confidence": pe.doppler_confidence,
                "doppler_motion": pe.doppler_motion,
                "frequency_r2": pe.frequency_r2,
                "tone_ratio": pe.tone_ratio,
                "decisions": {name: asdict(d) for name, d in decisions.items()},
            }
        )
    return rows


def _metrics(rows: list[dict], method: str) -> dict:
    truth_count = Counter(r["truth"] for r in rows)
    correct = Counter()
    confusion = defaultdict(Counter)
    unknown = 0
    speed_correct = speed_total = 0
    for row in rows:
        truth = row["truth"]
        decision = row["decisions"][method]
        pred = decision["motion"]
        confusion[truth][pred] += 1
        if pred == truth:
            correct[truth] += 1
        if pred == "unknown":
            unknown += 1
        if truth == "approaching" and pred == "approaching" and decision["speed_tier"] is not None:
            speed_total += 1
            speed_correct += int(decision["speed_tier"] == row["truth_speed_tier"])
    recall = {m: correct[m] / truth_count[m] if truth_count[m] else None for m in MOTIONS}
    available = [v for v in recall.values() if v is not None]
    balanced = float(np.mean(available)) if available else 0.0
    approach_recall = recall["approaching"] or 0.0
    recede_recall = recall["receding"] or 0.0
    steady_recall = recall["steady"] or 0.0
    safety_score = 0.5 * approach_recall + 0.25 * recede_recall + 0.25 * steady_recall
    return {
        "balanced_accuracy": balanced,
        "safety_score": safety_score,
        "recall": recall,
        "unknown_rate": unknown / len(rows) if rows else 0.0,
        "speed_tier_accuracy": speed_correct / speed_total if speed_total else None,
        "speed_tier_coverage": speed_total / truth_count["approaching"] if truth_count["approaching"] else None,
        "confusion": {t: dict(c) for t, c in confusion.items()},
        "ticks": len(rows),
    }


def run_benchmark(output: Path, limit_sources: int = 10) -> dict:
    scenarios = build_or_load_scenarios(limit_sources=limit_sources)
    rows: list[dict] = []
    for i, scenario in enumerate(scenarios, 1):
        print(f"[{i:02d}/{len(scenarios):02d}] {scenario.scenario_id}")
        rows.extend(_scenario_rows(scenario))
    methods = ("model", "acoustic", "doppler", "b", "c")
    metrics = {method: _metrics(rows, method) for method in methods}
    ranking = sorted(methods, key=lambda m: metrics[m]["safety_score"], reverse=True)
    result = {
        "description": "공개 실제 사이렌 원본 + pyroadacoustics 이동 시뮬레이션",
        "source_count": limit_sources,
        "scenario_count": len(scenarios),
        "ranking": ranking,
        "metrics": metrics,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nmethod       safety balanced  appr   recede steady unknown speed_acc coverage")
    for method in ranking:
        m = metrics[method]
        r = m["recall"]
        sp = m["speed_tier_accuracy"]
        cov = m["speed_tier_coverage"]
        print(
            f"{method:10s} {m['safety_score']:.3f}  {m['balanced_accuracy']:.3f}  "
            f"{(r['approaching'] or 0):.3f}  {(r['receding'] or 0):.3f}  {(r['steady'] or 0):.3f}  "
            f"{m['unknown_rate']:.3f}  {sp if sp is not None else float('nan'):.3f}  "
            f"{cov if cov is not None else float('nan'):.3f}"
        )
    print(f"\nresult: {output}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/evaluation_results.json"))
    parser.add_argument("--sources", type=int, default=10)
    args = parser.parse_args()
    run_benchmark(args.output, args.sources)


if __name__ == "__main__":
    main()
