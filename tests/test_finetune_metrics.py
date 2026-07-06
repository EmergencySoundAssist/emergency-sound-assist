"""한국어 CER 정규화·키워드 히트 판정 테스트."""
from finetune.metrics import normalize_ko, keyword_hit


def test_normalize_strips_punct_and_space():
    assert normalize_ko("차 세우세요!") == "차세우세요"
    assert normalize_ko("  안 돼요...  ") == "안돼요"
    assert normalize_ko("STOP! 멈춰요") == "stop멈춰요"


def test_normalize_empty_and_none():
    assert normalize_ko("") == ""
    assert normalize_ko(None) == ""
    assert normalize_ko("!?.,~") == ""


def test_normalize_nfc():
    # NFD(자모 분해) 입력도 NFC 로 합쳐져 같은 결과가 나와야 한다.
    import unicodedata
    nfd = unicodedata.normalize("NFD", "구급차")
    assert normalize_ko(nfd) == "구급차"


def test_keyword_hit_all_required():
    assert keyword_hit(["구급차"], "지금 구급차가 지나갑니다") is True
    assert keyword_hit(["음주", "단속"], "음주 단속 중입니다") is True
    assert keyword_hit(["음주", "단속"], "음주 운전은 위험합니다") is False
    assert keyword_hit(["안돼"], "안 돼요") is True      # 공백 무시 매칭


def test_keyword_hit_empty_hyp():
    assert keyword_hit(["구급차"], "") is False
