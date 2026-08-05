"""실제 faster-whisper로 한국어 음성+공개 도로소음 STT 정확도와 지연을 측정한다.

macOS ``say`` TTS 또는 FLEURS 한국어 실제 음성에 Figshare 도로소음을 지정 SNR로 섞고,
제품과 동일한 1초 입력·Silero VAD·발화 버퍼 경로를 탄다.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from core.types import AudioChunk, SAMPLE_RATE
from stt.config import STTConfig
from stt.transcriber import FasterWhisperEngine, Transcriber


TEXTS = (
    "앞에 사고가 발생했습니다",
    "차를 오른쪽에 세워 주세요",
    "구급차가 지나갑니다",
    "위험하니 천천히 운전하세요",
    "도로 공사 중입니다",
    "전방에 긴급 차량이 접근하고 있습니다",
    "좌측에서 구급차가 오고 있습니다",
    "경찰차가 뒤에서 접근합니다",
    "소방차에게 길을 양보해 주세요",
    "교차로에서 정지해 주세요",
)


def _resample(x: np.ndarray, source_sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    if source_sr == target_sr:
        return x.astype(np.float32, copy=False)
    import math

    gcd = math.gcd(int(source_sr), int(target_sr))
    return resample_poly(x, target_sr // gcd, source_sr // gcd).astype(np.float32)


def _read(path: Path) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return _resample(np.asarray(x, dtype=np.float32), int(sr))


def generate_tts(output_dir: Path, voices: tuple[str, ...]) -> list[tuple[str, str, np.ndarray]]:
    """macOS say로 정답 문장이 알려진 평가 음성을 만든다."""
    if subprocess.run(["which", "say"], capture_output=True).returncode != 0:
        raise RuntimeError("이 평가의 TTS 생성에는 macOS say가 필요합니다")
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus = []
    for voice in voices:
        for index, text in enumerate(TEXTS, 1):
            stem = f"{voice.lower()}-{index}"
            aiff = output_dir / f"{stem}.aiff"
            wav = output_dir / f"{stem}.wav"
            if not wav.exists():
                subprocess.run(
                    ["say", "-v", voice, "-r", "175", "-o", str(aiff), text],
                    check=True,
                )
                samples = _read(aiff)
                sf.write(wav, samples, SAMPLE_RATE, subtype="PCM_16")
                aiff.unlink(missing_ok=True)
            samples = _read(wav)
            if samples.size < SAMPLE_RATE // 2 or not np.any(samples):
                raise RuntimeError(f"TTS 음성이 비어 있습니다. 설치된 한국어 voice를 확인하세요: {voice}")
            corpus.append((stem, text, samples))
    return corpus


def load_fleurs(manifest_path: Path) -> list[tuple[str, str, np.ndarray]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = manifest_path.parent
    return [
        (Path(row["path"]).stem, row["reference"], _read(base / row["path"]))
        for row in manifest["samples"]
    ]


def _loop_noise(noise: np.ndarray, length: int, offset: int) -> np.ndarray:
    if not noise.size:
        return np.zeros(length, dtype=np.float32)
    start = offset % len(noise)
    indices = (start + np.arange(length)) % len(noise)
    return noise[indices].astype(np.float32)


def _mix(speech: np.ndarray, noise: np.ndarray, snr_db: float | None, seed: int):
    pad = int(0.25 * SAMPLE_RATE)
    clean = np.pad(speech, (pad, pad)).astype(np.float32)
    if snr_db is None:
        return clean, np.zeros(SAMPLE_RATE, dtype=np.float32)
    rng = np.random.default_rng(seed)
    offset = int(rng.integers(0, max(1, len(noise))))
    background = _loop_noise(noise, len(clean) + SAMPLE_RATE, offset)
    speech_rms = float(np.sqrt(np.mean(speech.astype(np.float64) ** 2))) + 1e-12
    noise_rms = float(np.sqrt(np.mean(background.astype(np.float64) ** 2))) + 1e-12
    scale = speech_rms / (10 ** (snr_db / 20.0) * noise_rms)
    mixed = clean + background[: len(clean)] * scale
    tail = background[len(clean) :] * scale
    peak = max(float(np.max(np.abs(mixed))), float(np.max(np.abs(tail))), 1.0)
    return (mixed / peak).astype(np.float32), (tail / peak).astype(np.float32)


def _normalize_text(text: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]", "", text.lower())


def _distance(a: list[str] | str, b: list[str] | str) -> int:
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def _scores(reference: str, hypothesis: str) -> tuple[float, float]:
    ref_chars = _normalize_text(reference)
    hyp_chars = _normalize_text(hypothesis)
    cer = _distance(ref_chars, hyp_chars) / max(1, len(ref_chars))
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    wer = _distance(ref_words, hyp_words) / max(1, len(ref_words))
    return float(cer), float(wer)


def _transcribe_stream(engine, cfg: STTConfig, audio: np.ndarray, tail: np.ndarray) -> dict:
    transcriber = Transcriber(config=cfg, engine=engine)
    result = None
    started = time.perf_counter()
    stream = np.concatenate([audio, tail])
    for start in range(0, len(stream), SAMPLE_RATE):
        chunk = stream[start : start + SAMPLE_RATE]
        if len(chunk) < SAMPLE_RATE:
            chunk = np.pad(chunk, (0, SAMPLE_RATE - len(chunk)))
        candidate = transcriber.transcribe(AudioChunk(chunk, SAMPLE_RATE))
        if candidate.text:
            result = candidate
    released = result is not None
    if result is None:
        candidate = transcriber.flush()
        result = candidate if candidate.text else None
    elapsed = time.perf_counter() - started
    return {
        "text": result.text if result else "",
        "confidence": result.confidence if result else 0.0,
        "released_by_vad": released,
        "elapsed_seconds": elapsed,
        "audio_seconds": len(audio) / SAMPLE_RATE,
    }


class _CountingEngine:
    """실제 엔진을 감싸 VAD가 엔진 호출까지 허용했는지 센다."""

    def __init__(self, engine):
        self.engine = engine
        self.calls = 0

    def transcribe(self, samples: np.ndarray, sample_rate: int):
        self.calls += 1
        return self.engine.transcribe(samples, sample_rate)


def _benchmark_noise_only(engine, cfg: STTConfig, noise: np.ndarray, seconds: int = 8) -> dict:
    """도로소음만 들어올 때 VAD 통과와 잘못된 자막 출력을 따로 측정한다."""
    counted = _CountingEngine(engine)
    transcriber = Transcriber(config=cfg, engine=counted)
    text_parts = []
    confidences = []
    started = time.perf_counter()
    stream = _loop_noise(noise, seconds * SAMPLE_RATE, offset=0)
    for start in range(0, len(stream), SAMPLE_RATE):
        result = transcriber.transcribe(
            AudioChunk(stream[start : start + SAMPLE_RATE], SAMPLE_RATE)
        )
        if result.text:
            text_parts.append(result.text)
            confidences.append(result.confidence)
    # 실제 스트림의 일시적인 정숙 구간을 모사해 남은 발화 버퍼의 VAD release를 유도한다.
    result = transcriber.transcribe(
        AudioChunk(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
    )
    if result.text:
        text_parts.append(result.text)
        confidences.append(result.confidence)
    result = transcriber.flush()
    if result.text:
        text_parts.append(result.text)
        confidences.append(result.confidence)
    return {
        "seconds": seconds,
        "engine_calls": counted.calls,
        "vad_triggered": counted.calls > 0,
        "text": " ".join(text_parts).strip(),
        "confidence": max(confidences, default=0.0),
        "false_caption": bool(text_parts),
        "elapsed_seconds": time.perf_counter() - started,
    }


def benchmark(
    model: str,
    voices: tuple[str, ...],
    snrs: tuple[float | None, ...],
    output: Path,
    road_dir: Path,
    corpus_kind: str = "tts",
    fleurs_manifest: Path = Path("data/public_fleurs_ko/manifest.json"),
) -> dict:
    if corpus_kind == "fleurs":
        if not fleurs_manifest.exists():
            raise RuntimeError(
                "FLEURS 표본이 없습니다. python -m evaluation.download_fleurs_samples"
            )
        corpus = load_fleurs(fleurs_manifest)
        description = "Google FLEURS ko_kr human speech + Figshare public road noise"
    else:
        corpus = generate_tts(Path("data/stt_tts"), voices)
        description = "macOS Korean TTS + Figshare public road noise; no real microphone"
    road_files = sorted(road_dir.glob("*.wav"))
    if not road_files:
        raise RuntimeError(
            "도로소음이 없습니다. python -m evaluation.download_figshare_samples --kind road --count 6"
        )
    noises = [_read(path) for path in road_files]
    cfg = STTConfig(
        model_size=model,
        device="cpu",
        compute_type="int8",
        vad_backend="silero",
        normalize_audio=False,
        beam_size=1,
    )
    load_start = time.perf_counter()
    engine = FasterWhisperEngine(cfg)
    model_load_seconds = time.perf_counter() - load_start

    cases = []
    for index, (sample_id, reference, speech) in enumerate(corpus):
        for snr in snrs:
            noise = noises[index % len(noises)]
            audio, tail = _mix(speech, noise, snr, seed=index * 100 + int(snr or 99))
            row = _transcribe_stream(engine, cfg, audio, tail)
            cer, wer = _scores(reference, row["text"])
            cases.append({
                "sample_id": sample_id,
                "reference": reference,
                "snr_db": snr,
                "cer": cer,
                "wer": wer,
                **row,
            })
            label = "clean" if snr is None else f"{snr:g}dB"
            print(f"[{len(cases):02d}] {sample_id} {label}: {row['text']!r} CER={cer:.2f}")

    noise_only = []
    for path, noise in zip(road_files, noises):
        row = _benchmark_noise_only(engine, cfg, noise)
        row["sample_id"] = path.stem
        noise_only.append(row)
        print(
            f"[noise] {path.stem}: calls={row['engine_calls']} "
            f"caption={row['text']!r}"
        )

    grouped = defaultdict(list)
    for row in cases:
        grouped["clean" if row["snr_db"] is None else f"{row['snr_db']:g}dB"].append(row)
    summary = {
        label: {
            "cases": len(rows),
            "mean_cer": float(np.mean([row["cer"] for row in rows])),
            "mean_wer": float(np.mean([row["wer"] for row in rows])),
            "exact_match_rate": float(np.mean([row["cer"] == 0 for row in rows])),
            "vad_release_rate": float(np.mean([row["released_by_vad"] for row in rows])),
            "mean_latency_seconds": float(np.mean([row["elapsed_seconds"] for row in rows])),
            "mean_realtime_factor": float(np.mean([
                row["elapsed_seconds"] / max(row["audio_seconds"], 1e-9) for row in rows
            ])),
        }
        for label, rows in grouped.items()
    }
    result = {
        "description": description,
        "corpus": corpus_kind,
        "model": model,
        "voices": voices,
        "model_load_seconds": model_load_seconds,
        "summary": summary,
        "noise_only_summary": {
            "cases": len(noise_only),
            "vad_trigger_rate": float(np.mean([row["vad_triggered"] for row in noise_only])),
            "false_caption_rate": float(np.mean([row["false_caption"] for row in noise_only])),
            "total_engine_calls": int(sum(row["engine_calls"] for row in noise_only)),
        },
        "cases": cases,
        "noise_only_cases": noise_only,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(result["noise_only_summary"], ensure_ascii=False, indent=2))
    print(f"result: {output}")
    return result


def _parse_snr(value: str) -> tuple[float | None, ...]:
    return tuple(None if part.strip().lower() == "clean" else float(part) for part in value.split(","))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="small")
    parser.add_argument("--corpus", choices=("tts", "fleurs"), default="tts")
    parser.add_argument("--voices", default="Yuna")
    parser.add_argument(
        "--fleurs-manifest", type=Path, default=Path("data/public_fleurs_ko/manifest.json")
    )
    parser.add_argument("--snr", default="clean,20,10,5")
    parser.add_argument("--road-dir", type=Path, default=Path("data/public_road_noise"))
    parser.add_argument("--output", type=Path, default=Path("data/stt_evaluation_results.json"))
    args = parser.parse_args()
    benchmark(
        args.model,
        tuple(part.strip() for part in args.voices.split(",") if part.strip()),
        _parse_snr(args.snr),
        args.output,
        args.road_dir,
        args.corpus,
        args.fleurs_manifest,
    )


if __name__ == "__main__":
    main()
