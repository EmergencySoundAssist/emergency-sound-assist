"""finetune 긴급문구 CSV 로더 테스트."""
from pathlib import Path

from finetune.phrases import load_phrases, CATEGORIES

CSV = Path("finetune/emergency_phrases.csv")


def test_csv_exists_and_loads():
    rows = load_phrases(CSV)
    assert len(rows) >= 75


def test_row_shape():
    rows = load_phrases(CSV)
    for r in rows:
        assert r["text"].strip(), r
        assert r["category"] in CATEGORIES, r
        assert isinstance(r["keywords"], list) and len(r["keywords"]) >= 1, r
        assert all(k.strip() for k in r["keywords"]), r


def test_keywords_are_substrings_of_normalized_text():
    # 키워드는 정규화된 원문에 부분문자열로 들어있어야 히트 판정이 성립한다.
    import re, unicodedata
    def norm(t):
        return re.sub(r"[^가-힣0-9a-z]", "", unicodedata.normalize("NFC", t).lower())
    rows = load_phrases(CSV)
    for r in rows:
        for k in r["keywords"]:
            assert norm(k) in norm(r["text"]), (r["text"], k)


def test_all_categories_covered():
    rows = load_phrases(CSV)
    assert {r["category"] for r in rows} == CATEGORIES
