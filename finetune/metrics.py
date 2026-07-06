"""평가 지표용 텍스트 처리 — 한국어 정규화·키워드 히트.

정규화 규칙(고정, 리포트에 명기됨):
  NFC → lowercase → 한글/숫자/영소문자 외 전부 제거(공백 포함).
한국어는 띄어쓰기가 유동적이라 공백 불문 CER 이 관례
(공백 유지 CER 은 띄어쓰기 차이를 오류로 세어 모델 간 비교를 흐린다).
숫자 표기(10↔십)는 페널티로 남음 — 알려진 한계로 문서화.
"""
from __future__ import annotations

import re
import unicodedata

_DROP = re.compile(r"[^가-힣0-9a-z]")


def normalize_ko(text) -> str:
    t = unicodedata.normalize("NFC", text or "").lower()
    return _DROP.sub("", t)


def keyword_hit(keywords: list[str], hyp_text: str) -> bool:
    """정규화 후 모든 키워드가 hyp 에 부분문자열로 존재하면 히트."""
    h = normalize_ko(hyp_text)
    ks = [normalize_ko(k) for k in keywords if k]
    return bool(ks) and all(k in h for k in ks)
