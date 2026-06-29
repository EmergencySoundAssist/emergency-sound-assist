"""
긴급 키워드 스포팅 (순수 함수 — 엔진·하드웨어 의존 없음, 테스트 쉬움).

인식 텍스트에서 긴급 키워드를 찾는다. STT 는 띄어쓰기·활용·1글자 오인식이 흔하므로:
  1) 공백/문장부호 제거 후 부분 문자열 매칭(빠른 1차)
  2) 안 잡히면 '자모(초·중·종성) 단위' 편집유사도로 fuzzy 매칭
     예) "구금차"→"구급차", "비켜 주세요"→"비켜" 처럼 살짝 틀려도 알림은 살린다.

자모 분해는 외부 의존성 없이 유니코드 한글 조합식으로 직접 계산한다.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Union

from .config import EMERGENCY_KEYWORDS

# 유니코드 한글 음절 분해용 자모 테이블
_CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_JONG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ",
         "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ",
         "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]


def _compact(s: str) -> str:
    """공백·문장부호 제거 + 소문자화 (띄어쓰기/부호 차이를 무시)."""
    return re.sub(r"[\s\W_]+", "", s, flags=re.UNICODE).lower()


def _jamo(s: str) -> str:
    """한글 음절을 초·중·종성으로 풀어쓴 문자열. (비한글은 그대로)"""
    out: List[str] = []
    for ch in s:
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:           # 완성형 한글
            idx = code - 0xAC00
            out.append(_CHO[idx // 588])
            out.append(_JUNG[(idx % 588) // 28])
            jong = idx % 28
            if jong:
                out.append(_JONG[jong])
        else:
            out.append(ch)
    return "".join(out)


def _fuzzy_hit(text_jamo: str, kw_jamo: str) -> bool:
    """text_jamo 안에 kw_jamo 와 충분히 비슷한 구간이 있나 (슬라이딩 윈도우)."""
    n = len(kw_jamo)
    if n == 0:
        return False
    # 짧은 키워드일수록 엄격(오경보 방지). 긴 건 약간 느슨.
    threshold = 0.85 if n <= 6 else 0.80
    for w in range(max(1, n - 2), n + 3):
        for i in range(0, len(text_jamo) - w + 1):
            if SequenceMatcher(None, text_jamo[i:i + w], kw_jamo).ratio() >= threshold:
                return True
    return False


def _as_groups(keywords) -> Dict[str, List[str]]:
    """list 가 와도 dict(대표어→[변형])로 정규화. (리스트면 각 항목이 곧 대표어)"""
    if keywords is None:
        return EMERGENCY_KEYWORDS
    if isinstance(keywords, dict):
        return keywords
    return {k: [k] for k in keywords}


def find_keywords(
    text: str,
    keywords: Optional[Union[Dict[str, List[str]], List[str]]] = None,
    fuzzy: bool = True,
) -> List[str]:
    """text 에서 긴급 '대표어'를 (정의 순서·중복 제거) 반환.

    대표어 하나라도 변형이 잡히면 그 대표어를 넣는다(변형 자체가 아니라).
    fuzzy=False 면 부분 문자열 매칭만(빠름).
    """
    if not text:
        return []
    groups = _as_groups(keywords)
    compact = _compact(text)
    text_jamo = _jamo(compact)

    found: List[str] = []
    for canonical, variants in groups.items():
        for var in variants:
            vc = _compact(var)
            if not vc:
                continue
            hit = vc in compact
            if not hit and fuzzy:
                hit = _fuzzy_hit(text_jamo, _jamo(vc))
            if hit:
                found.append(canonical)
                break          # 이 대표어는 잡혔으니 다음 대표어로
    return found
