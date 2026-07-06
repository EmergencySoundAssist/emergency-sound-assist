"""공개 한국어 코퍼스 test split 에서 평가용 발화 샘플링 (네트워크·HF 캐시 사용).

- zeroth-korean test 457발화(16k, CC-BY-4.0, 비게이트) 중 130
- google/fleurs ko_kr test 382발화(16k, CC-BY-4.0, 비게이트) 중 70
  → 총 200발화 ≈ 30분. train split 은 건드리지 않는다(미래 학습셋과 화자 분리).

⚠️ Common Voice ko 는 2025-10 부로 HF 에서 데이터가 내려가 제외(스펙 편차 —
   docs/stt/finetune.md 기록). ⚠️ torch/torchcodec 불필요: Audio(decode=False) 로
   bytes 만 받아 soundfile 로 직접 디코딩한다.

실행: python3 -m finetune.corpora   # data/eval_v0/corpus/*.wav + raw_corpus.jsonl
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from . import SEED
from .audio_io import SR, save_wav_16k

OUT_DIR = Path("data/eval_v0/corpus")
MANIFEST = Path("data/eval_v0/raw_corpus.jsonl")

#           (source,  repo,                    config,  text컬럼,        개수, 시드오프셋)
_SOURCES = [("zeroth", "kresnik/zeroth_korean", None,    "text",          130,  0),
            ("fleurs", "google/fleurs",         "ko_kr", "transcription",  70,  1)]


def select_indices(n_total: int, k: int, seed: int) -> list[int]:
    """비복원 결정적 샘플 — 같은 (n_total, k, seed) 면 항상 같은 결과."""
    rng = np.random.default_rng(seed)
    k = min(k, n_total)
    return sorted(int(i) for i in rng.choice(n_total, size=k, replace=False))


def _decode_16k(audio_cell: dict) -> np.ndarray:
    """datasets Audio(decode=False) 셀({'bytes','path'}) → 16k mono float32."""
    if audio_cell.get("bytes"):
        data, sr = sf.read(io.BytesIO(audio_cell["bytes"]), dtype="float32")
    else:
        data, sr = sf.read(audio_cell["path"], dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SR:
        data = soxr.resample(data, sr, SR)
    return np.asarray(data, dtype=np.float32)


def main() -> None:
    from datasets import Audio, load_dataset      # 지연 import(무거움)

    entries = []
    for source, repo, config, text_col, k, seed_off in _SOURCES:
        print(f"[corpora] 로드: {repo} ({config or 'default'}) test split …")
        ds = load_dataset(repo, config, split="test") if config else \
             load_dataset(repo, split="test")
        ds = ds.cast_column("audio", Audio(decode=False))
        # 빈 전사 방어: 후보를 k+10 뽑아 비어있지 않은 것 k개를 채운다(결정적).
        cand = select_indices(len(ds), min(k + 10, len(ds)), SEED + seed_off)
        taken = 0
        for i in cand:
            if taken >= k:
                break
            row = ds[i]
            text = (row[text_col] or "").strip()
            if not text:
                continue
            uid = f"corpus_{source}_{i:05d}"
            wav_path = OUT_DIR / f"{uid}.wav"
            if not wav_path.exists():
                save_wav_16k(wav_path, _decode_16k(row["audio"]))
            entries.append({"id": uid, "path": str(wav_path),
                            "text": text, "source": source})
            taken += 1
        print(f"[corpora] {source}: {taken}건 (목표 {k})")
        if taken < k:
            print(f"[corpora] ⚠️ {source}: 빈 전사 과다로 {k - taken}건 부족",
                  file=sys.stderr)

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"[corpora] 완료: {len(entries)}건 → {MANIFEST}")


if __name__ == "__main__":
    main()
