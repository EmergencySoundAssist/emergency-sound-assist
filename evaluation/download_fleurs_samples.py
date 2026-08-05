"""Google FLEURS 한국어 dev에서 재현 가능한 소규모 실제 음성 평가 표본을 추출한다."""

from __future__ import annotations

import argparse
import csv
import json
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "google/fleurs"
TSV_NAME = "data/ko_kr/dev.tsv"
ARCHIVE_NAME = "data/ko_kr/audio/dev.tar.gz"


def _rows(tsv_path: Path, max_seconds: float) -> list[dict]:
    rows = []
    with tsv_path.open(encoding="utf-8", newline="") as handle:
        for fields in csv.reader(handle, delimiter="\t"):
            if len(fields) < 7:
                continue
            sentence_id, filename, raw, normalized, _, sample_count, gender = fields[:7]
            seconds = int(sample_count) / 16000
            if 0.5 <= seconds <= max_seconds:
                rows.append({
                    "sentence_id": sentence_id,
                    "filename": filename,
                    "reference": normalized,
                    "raw_reference": raw,
                    "seconds": seconds,
                    "gender": gender,
                })
    return rows


def _select(rows: list[dict], count: int) -> list[dict]:
    """남녀를 번갈아 고르며 같은 원문 문장의 중복 녹음을 피한다."""
    buckets = {
        gender: [row for row in rows if row["gender"] == gender]
        for gender in ("FEMALE", "MALE")
    }
    selected = []
    seen_sentence_ids = set()
    seen_references = set()
    while len(selected) < count:
        progressed = False
        for gender in ("FEMALE", "MALE"):
            for row in buckets[gender]:
                if row["sentence_id"] in seen_sentence_ids:
                    continue
                if row["reference"] in seen_references:
                    continue
                selected.append(row)
                seen_sentence_ids.add(row["sentence_id"])
                seen_references.add(row["reference"])
                progressed = True
                break
            if len(selected) >= count:
                break
        if not progressed:
            break
    if len(selected) < count:
        raise RuntimeError(f"조건을 만족하는 표본은 {len(selected)}개뿐입니다")
    return selected


def download(output_dir: Path, count: int, max_seconds: float) -> Path:
    source_dir = output_dir.parent / f"{output_dir.name}_source"
    tsv = Path(hf_hub_download(
        repo_id=REPO_ID, repo_type="dataset", filename=TSV_NAME,
        local_dir=source_dir,
    ))
    archive = Path(hf_hub_download(
        repo_id=REPO_ID, repo_type="dataset", filename=ARCHIVE_NAME,
        local_dir=source_dir,
    ))
    selected = _select(_rows(tsv, max_seconds), count)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as bundle:
        for row in selected:
            member = bundle.getmember(f"dev/{row['filename']}")
            source = bundle.extractfile(member)
            if source is None:
                raise RuntimeError(f"압축 파일을 읽지 못했습니다: {member.name}")
            target = output_dir / row["filename"]
            target.write_bytes(source.read())
            row["path"] = target.name

    manifest = {
        "dataset": "Google FLEURS ko_kr dev",
        "source": "https://huggingface.co/datasets/google/fleurs",
        "license": "CC BY 4.0",
        "selection": "alternating gender; unique sentence/reference; duration <= max_seconds",
        "count": len(selected),
        "samples": selected,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"FLEURS 한국어 표본 {len(selected)}개: {output_dir}")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/public_fleurs_ko"))
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--max-seconds", type=float, default=7.5)
    args = parser.parse_args()
    download(args.output_dir, args.count, args.max_seconds)


if __name__ == "__main__":
    main()
