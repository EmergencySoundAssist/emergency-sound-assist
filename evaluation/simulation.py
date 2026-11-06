"""공개 사이렌 원본으로 정답이 있는 이동/정지 평가 시나리오를 만든다."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


TARGET_SR = 16_000
SIM_SR = 8_000
TICK_SECONDS = 0.5


@dataclass(frozen=True)
class SourceClip:
    source_id: str
    samples: np.ndarray
    sample_rate: int


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    source_id: str
    kind: str
    samples: np.ndarray
    sample_rate: int
    pass_time: float | None
    speed_tier: int | None  # 1=느림, 2=보통, 3=빠름
    duration: float

    def truth_at(self, time_seconds: float) -> str:
        if self.kind.startswith("static"):
            return "steady"
        if self.kind.startswith("passby"):
            assert self.pass_time is not None
            dead = 0.25
            if time_seconds < self.pass_time - dead:
                return "approaching"
            if time_seconds > self.pass_time + dead:
                return "receding"
            return "steady"
        raise ValueError(f"알 수 없는 scenario kind={self.kind}")


def _float_mono(samples: np.ndarray) -> np.ndarray:
    x = np.asarray(samples)
    if x.ndim == 2:
        x = x.astype(np.float64).mean(axis=1)
    else:
        x = x.astype(np.float64)
    if np.issubdtype(samples.dtype, np.integer):
        info = np.iinfo(samples.dtype)
        scale = max(abs(info.min), abs(info.max))
        x /= float(scale)
    x -= float(np.mean(x))
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 0:
        x = 0.8 * x / peak
    return x.astype(np.float32)


def load_public_sources(
    data_dir: Path = Path("data/public_sirens"),
    min_duration: float = 1.5,
    limit: int = 10,
) -> list[SourceClip]:
    """길이가 충분한 실제 사이렌 WAV를 서로 다른 원본으로 불러온다."""
    clips: list[SourceClip] = []
    for path in sorted(data_dir.glob("*.wav")):
        sr, raw = wavfile.read(path)
        if len(raw) / sr < min_duration:
            continue
        x = _float_mono(raw)
        clips.append(SourceClip(path.stem, x, int(sr)))
    if len(clips) < min(2, limit):
        raise RuntimeError(
            f"평가용 공개 WAV가 부족합니다({len(clips)}개). "
            "python -m evaluation.download_figshare_samples 를 먼저 실행하세요."
        )
    return clips[:limit]


def _resample(x: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return x.astype(np.float32, copy=False)
    import math

    gcd = math.gcd(int(source_sr), int(target_sr))
    return resample_poly(x, target_sr // gcd, source_sr // gcd).astype(np.float32)


def _loop_crossfade(x: np.ndarray, length: int, fade_samples: int) -> np.ndarray:
    """짧은 공개 음원을 필요한 길이까지 반복하되 경계 클릭을 완화한다."""
    if x.size == 0:
        return np.zeros(length, dtype=np.float32)
    if len(x) >= length:
        return x[:length].astype(np.float32, copy=True)
    out = np.empty(0, dtype=np.float32)
    while len(out) < length:
        block = x.astype(np.float32, copy=True)
        if out.size and fade_samples > 0:
            n = min(fade_samples, len(out), len(block))
            w = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)
            out[-n:] = out[-n:] * (1.0 - w) + block[:n] * w
            block = block[n:]
        out = np.concatenate([out, block])
    return out[:length]


def _normalize_rms(x: np.ndarray, target: float = 0.08) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0
    if rms <= 1e-8:
        return x.astype(np.float32)
    return np.clip(x * (target / rms), -1.0, 1.0).astype(np.float32)


def _add_noise(x: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(len(x))
    sp = float(np.mean(np.square(x, dtype=np.float64))) + 1e-12
    npow = float(np.mean(noise**2)) + 1e-12
    noise *= np.sqrt(sp / (10.0 ** (snr_db / 10.0)) / npow)
    return np.clip(x + noise, -1.0, 1.0).astype(np.float32)


def _seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")


def static_scenario(
    source: SourceClip,
    duration: float = 16.0,
    gain_mode: str = "constant",
    snr_db: float = 20.0,
) -> Scenario:
    x = _resample(source.samples, source.sample_rate, TARGET_SR)
    n = round(duration * TARGET_SR)
    x = _loop_crossfade(x, n, round(0.03 * TARGET_SR))
    if gain_mode == "rising":
        gain = np.exp(np.linspace(np.log(0.25), np.log(1.0), n))
    elif gain_mode == "falling":
        gain = np.exp(np.linspace(np.log(1.0), np.log(0.25), n))
    elif gain_mode == "constant":
        gain = np.ones(n)
    else:
        raise ValueError(gain_mode)
    x = _normalize_rms((x * gain).astype(np.float32))
    sid = f"{source.source_id}-static-{gain_mode}"
    x = _add_noise(x, snr_db, _seed(sid))
    return Scenario(sid, source.source_id, f"static_{gain_mode}", x, TARGET_SR, None, None, duration)


def passby_scenario(
    source: SourceClip,
    speed_kmh: float,
    lateral_m: float = 8.0,
    duration: float = 16.0,
    snr_db: float = 20.0,
    include_reflection: bool = True,
) -> Scenario:
    """pyroadacoustics로 도플러·거리감쇠·노면반사가 있는 통과 신호 생성."""
    # pyroadacoustics는 8 kHz에서 충분하고 훨씬 빠르다. 생성 후 시스템 입력 16 kHz로 변환한다.
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/emergency-sound-assist-mpl")
    import pyroadacoustics as pra

    src = _resample(source.samples, source.sample_rate, SIM_SR)
    src = _loop_crossfade(src, round(duration * SIM_SR), round(0.03 * SIM_SR))
    src = _normalize_rms(src, 0.15)
    v = float(speed_kmh) / 3.6
    half_y = v * duration / 2.0
    height = 1.2
    trajectory = np.array(
        [[lateral_m, -half_y, height], [lateral_m, half_y, height]], dtype=float
    )
    env = pra.Environment(fs=SIM_SR, temperature=20, pressure=1, rel_humidity=50)
    env.set_simulation_params(
        interp_method="Linear",
        include_reflection=include_reflection,
        include_air_absorption=False,
    )
    env.add_source(
        trajectory[0],
        signal=src,
        trajectory_points=trajectory,
        source_velocity=np.array([v]),
    )
    env.add_microphone_array(np.array([[0.0, 0.0, height]]))
    received = np.asarray(env.simulate()[0], dtype=np.float32)
    received = _resample(received, SIM_SR, TARGET_SR)
    wanted = round(duration * TARGET_SR)
    if len(received) < wanted:
        received = np.pad(received, (0, wanted - len(received)))
    received = _normalize_rms(received[:wanted])
    tier = 1 if speed_kmh < 30 else 2 if speed_kmh < 50 else 3
    sid = f"{source.source_id}-passby-v2-{int(speed_kmh)}-{int(lateral_m)}m"
    received = _add_noise(received, snr_db, _seed(sid))
    return Scenario(
        sid,
        source.source_id,
        "passby",
        received,
        TARGET_SR,
        duration / 2.0,
        tier,
        duration,
    )


def scenario_matrix(
    sources: list[SourceClip],
    speeds_kmh: tuple[int, ...] = (20, 40, 60),
) -> Iterator[tuple[SourceClip, str, dict]]:
    """캐시 생성용 시나리오 정의를 순서대로 제공한다."""
    for source in sources:
        for gain_mode in ("constant", "rising", "falling"):
            yield source, "static", {"gain_mode": gain_mode}
        for speed in speeds_kmh:
            yield source, "passby", {"speed_kmh": speed}


def build_or_load_scenarios(
    cache_dir: Path = Path("data/evaluation_cache"),
    source_dir: Path = Path("data/public_sirens"),
    limit_sources: int = 10,
) -> list[Scenario]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    sources = load_public_sources(source_dir, limit=limit_sources)
    scenarios: list[Scenario] = []
    for source, kind, kwargs in scenario_matrix(sources):
        if kind == "static":
            scenario = static_scenario(source, **kwargs)
        else:
            # passby는 비싸므로 먼저 캐시 유무를 예상 ID로 확인한다.
            speed = int(kwargs["speed_kmh"])
            expected_id = f"{source.source_id}-passby-v2-{speed}-8m"
            npz_path = cache_dir / f"{expected_id}.npz"
            meta_path = cache_dir / f"{expected_id}.json"
            if npz_path.exists() and meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                samples = np.load(npz_path)["samples"].astype(np.float32)
                scenario = Scenario(samples=samples, **meta)
            else:
                scenario = passby_scenario(source, **kwargs)
        npz_path = cache_dir / f"{scenario.scenario_id}.npz"
        meta_path = cache_dir / f"{scenario.scenario_id}.json"
        if not npz_path.exists():
            np.savez_compressed(npz_path, samples=scenario.samples)
            meta = {
                "scenario_id": scenario.scenario_id,
                "source_id": scenario.source_id,
                "kind": scenario.kind,
                "sample_rate": scenario.sample_rate,
                "pass_time": scenario.pass_time,
                "speed_tier": scenario.speed_tier,
                "duration": scenario.duration,
            }
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        scenarios.append(scenario)
    return scenarios
