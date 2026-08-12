"""차종 떨림 억제 — 창 하나의 추정을 그대로 그리지 않는다.

현장 신고: "핸드폰으로 사이렌소리를 틀어도 이제 막 차종이 튀네".
원인은 검출이 빨라지면서(early3/early4) 사이렌이 멀고 약할 때부터 창마다 차종을
뽑게 된 것 — 그 창들에는 차종 근거가 거의 없어 매 초 답이 바뀐다.
"""

import numpy as np
import pytest

from core.types import AudioChunk, ClassResult, SirenSubtype, SoundClass
from pipeline.runner import Pipeline


def _siren(sub, conf=0.9):
    return ClassResult.from_label(SoundClass.SIREN, 0.95,
                                  subtype=sub, subtype_confidence=conf)


def _run(seq):
    """분류 결과 열을 파이프라인에 흘리고 화면에 나갈 차종 열을 돌려준다."""
    import pipeline.runner as R
    p = Pipeline(clock=lambda: 0.0)
    out = []
    for r in seq:
        out.append(p._vote_subtype(r).subtype)
    return out


def test_flapping_subtype_never_names_a_vehicle():
    """구급차↔경찰차로 번갈아 나오면 화면은 '긴급차량'(UNKNOWN)이어야 한다."""
    seq = [_siren(SirenSubtype.AMBULANCE), _siren(SirenSubtype.POLICE)] * 6
    assert set(_run(seq)) == {SirenSubtype.UNKNOWN}


def test_consistent_subtype_is_named_after_enough_votes():
    """같은 차종이 꾸준히 나오면 그때는 이름을 붙인다 — 억제이지 봉인이 아니다."""
    out = _run([_siren(SirenSubtype.FIRE)] * 8)
    assert out[0] is SirenSubtype.UNKNOWN                  # 표가 모자란 초반은 보류
    assert out[-1] is SirenSubtype.FIRE
    assert out.index(SirenSubtype.FIRE) >= Pipeline.SUBTYPE_MIN_VOTES - 1


def test_vote_resets_between_events():
    """앞 사이렌의 표가 다음 사이렌으로 넘어가면 안 된다."""
    p = Pipeline(clock=lambda: 0.0)
    for _ in range(8):
        p._vote_subtype(_siren(SirenSubtype.FIRE))
    p._vote_subtype(ClassResult.from_label(SoundClass.NORMAL_TRAFFIC, 0.9))   # 이벤트 종료
    got = p._vote_subtype(_siren(SirenSubtype.AMBULANCE))
    assert got.subtype is SirenSubtype.UNKNOWN


def test_low_confidence_ticks_abstain_rather_than_vote():
    """확신 미달 틱은 UNKNOWN 으로 오는데, 그게 표로 세어지면 안 된다(기권)."""
    seq = [_siren(SirenSubtype.UNKNOWN, 0.4)] * 10
    assert set(_run(seq)) == {SirenSubtype.UNKNOWN}


def test_horn_does_not_carry_a_vehicle_type():
    p = Pipeline(clock=lambda: 0.0)
    got = p._vote_subtype(ClassResult.from_label(SoundClass.HORN, 0.9))
    assert got.subtype is None
