"""
긴급 키워드 스포팅 (순수 함수 — 엔진·하드웨어 의존 없음, 테스트 쉬움).

인식된 텍스트에서 config.EMERGENCY_KEYWORDS 의 단어를 찾아 돌려준다.
부분 문자열 매칭이라 활용형 일부를 함께 잡는다(예: "비키" → "비키세요").
"""

from __future__ import annotations

from typing import List, Optional

from .config import EMERGENCY_KEYWORDS


def find_keywords(text: str, keywords: Optional[List[str]] = None) -> List[str]:
    """text 안에 등장하는 긴급 키워드를 (등장 순서·중복 제거하여) 반환.

    매칭은 대소문자 무시(영문 별칭 대비)·부분 문자열 기준.
    """
    if not text:
        return []
    kws = keywords if keywords is not None else EMERGENCY_KEYWORDS
    lowered = text.lower()

    found: List[str] = []
    for kw in kws:
        if kw.lower() in lowered and kw not in found:
            found.append(kw)
    return found
