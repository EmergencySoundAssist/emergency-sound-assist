"""긴급문구 스크립트 로더. CSV 컬럼: text,category,keywords(|구분)."""
from __future__ import annotations

import csv
from pathlib import Path

CATEGORIES = {"police", "rescue", "pedestrian", "announce", "shout"}


def load_phrases(csv_path) -> list[dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "text": r["text"].strip(),
                "category": r["category"].strip(),
                "keywords": [k.strip() for k in r["keywords"].split("|") if k.strip()],
            })
    return rows
