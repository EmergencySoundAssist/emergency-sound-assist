"""평가셋 v0 최종 조립 — 조건(clean/noise/loudspeaker) 적용 + manifest.jsonl.

배분(스펙 §1):
- corpus 200: clean/snr10/snr5/snr0 균등(각 50)
- tts 150: 절반(75) 확성기(+도로소음 snr 10/5/0 순환), 나머지 75 는 4조건 순환
시드 고정 → 항상 같은 조건·같은 노이즈 크롭 → 완전 재현.

실행: python3 -m finetune.build_eval   # data/eval_v0/{wav/,manifest.jsonl}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import SEED
from .audio_io import load_mono_16k, save_wav_16k, crop_or_tile
from .augment import mix_at_snr, simulate_loudspeaker
from .noise import list_noise_files

_SNR_CYCLE = [10, 5, 0]


def assign_conditions(n: int, kind: str, seed: int) -> list[tuple[str, int | None]]:
    """항목 i → (condition, snr_db). 분포 고정 + 시드 셔플 = 결정적."""
    base: list[tuple[str, int | None]] = []
    if kind == "tts":
        n_ls = (n + 1) // 2
        base += [("loudspeaker", _SNR_CYCLE[i % 3]) for i in range(n_ls)]
        rest = n - n_ls
    elif kind == "corpus":
        rest = n
    else:
        raise ValueError(f"kind: {kind}")
    four: list[tuple[str, int | None]] = [("clean", None), ("noise", 10),
                                          ("noise", 5), ("noise", 0)]
    base += [four[i % 4] for i in range(rest)]
    rng = np.random.default_rng(seed)
    rng.shuffle(base)
    return base


def _read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build(tts_manifest, corpus_manifest, noise_dir, out_dir, seed: int = SEED) -> int:
    out_dir = Path(out_dir)
    wav_dir = out_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    noise_files = list_noise_files(noise_dir)
    if not noise_files:
        raise SystemExit(f"노이즈 풀이 비어있음: {noise_dir} — 먼저 python3 -m finetune.noise")

    groups = [("tts", _read_jsonl(tts_manifest)), ("corpus", _read_jsonl(corpus_manifest))]
    rng = np.random.default_rng(seed)          # 노이즈 선택·크롭용(조건 셔플과 분리)
    rows_out = []
    for kind, items in groups:
        conds = assign_conditions(len(items), kind, seed)
        for item, (cond, snr) in zip(items, conds):
            x = load_mono_16k(item["path"])
            if cond == "loudspeaker":
                x = simulate_loudspeaker(x)
            if cond in ("noise", "loudspeaker"):
                nf = noise_files[int(rng.integers(0, len(noise_files)))]
                noise = crop_or_tile(load_mono_16k(nf), len(x), rng)
                x = mix_at_snr(x, noise, float(snr))
            dst = wav_dir / f"{item['id']}.wav"
            save_wav_16k(dst, x)
            rows_out.append({
                "id": item["id"], "path": str(dst),
                "ref_text": item["text"], "source": item["source"],
                "condition": cond, "snr_db": snr,
                "category": item.get("category"), "keywords": item.get("keywords"),
                "tts_voice": item.get("voice"),
            })

    manifest = out_dir / "manifest.jsonl"
    with open(manifest, "w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[build_eval] {len(rows_out)}건 → {manifest}")
    return len(rows_out)


def main() -> None:
    n = build(tts_manifest="data/eval_v0/raw_tts.jsonl",
              corpus_manifest="data/eval_v0/raw_corpus.jsonl",
              noise_dir="data/noise", out_dir="data/eval_v0")
    from collections import Counter
    rows = _read_jsonl("data/eval_v0/manifest.jsonl")
    print("[build_eval] 조건 분포:", dict(Counter(f"{r['condition']}@{r['snr_db']}"
                                                   for r in rows)))
    assert n == len(rows)


if __name__ == "__main__":
    main()
