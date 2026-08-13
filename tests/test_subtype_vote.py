"""차종 떨림 억제 — 창 하나의 추정을 그대로 그리지 않는다.

현장 신고: "핸드폰으로 사이렌소리를 틀어도 이제 막 차종이 튀네".
원인은 검출이 빨라지면서(early3/early4) 사이렌이 멀고 약할 때부터 창마다 차종을
뽑게 된 것 — 그 창들에는 차종 근거가 거의 없어 매 초 답이 바뀐다.

★ 이 파일은 **Pipeline.process() 전체 경로**로 잰다. _vote_subtype 을 직접 부르면
  디바운스를 건너뛰는데, 첫 판에서 놓친 결함이 정확히 거기 있었다 — 디바운스가 잔향
  동안 같은 결과 객체를 계속 돌려주는 걸 표로 세어, 창 하나가 최대 8표가 됐다.
"""

import numpy as np
import pytest

from core.types import AudioChunk, ClassResult, SirenSubtype, SoundClass
from pipeline.runner import Pipeline

SR = 16_000


def _siren(sub, conf=0.9):
    return ClassResult.from_label(SoundClass.SIREN, 0.95,
                                  subtype=sub, subtype_confidence=conf)


_NOISE = ClassResult.from_label(SoundClass.NORMAL_TRAFFIC, 0.9)


def _drive(monkeypatch, seq, grid=1.0, hangover=2.0):
    """분류 결과 열을 process() 로 흘리고 **화면에 나갈** 차종 열을 돌려준다.

    시계는 청크 길이만큼 흐르게 둔다 — 벽시계를 고정하면 잔향이 영원히 안 끝나
    실제 동작과 달라진다.
    """
    import pipeline.runner as R

    it = iter(seq)
    monkeypatch.setattr(R, "classify", lambda chunk: next(it))
    t = {"v": 0.0}

    def clock():
        return t["v"]

    p = Pipeline(clock=clock, emergency_hangover=hangover)
    out = []
    n = int(SR * grid)
    for _ in seq:
        out.append(p.process(AudioChunk(samples=np.zeros(n, np.float32),
                                        sample_rate=SR)).sound.subtype)
        t["v"] += grid
    return out


def test_hangover_replay_does_not_manufacture_votes(monkeypatch):
    """★ 회귀: 진짜 사이렌 창 1개 + 잔향. 창 하나로 차종을 확정하면 안 된다.

    디바운스는 잔향 동안 같은 ClassResult 를 되돌려준다. 그걸 표로 세면 관측 1개가
    표 5개를 만들어 '구급차'로 확정된다 — 이 함수가 막으려던 바로 그 일이다.
    """
    seq = [_siren(SirenSubtype.AMBULANCE, 0.62)] + [_NOISE] * 11
    out = _drive(monkeypatch, seq, grid=0.25, hangover=2.0)
    assert SirenSubtype.AMBULANCE not in out, f"관측 1개로 차종 확정됨: {out}"


def test_flapping_subtype_never_names_a_vehicle(monkeypatch):
    """구급차↔경찰차로 번갈아 나오면 화면은 '긴급차량'(UNKNOWN)이어야 한다."""
    seq = [_siren(SirenSubtype.AMBULANCE), _siren(SirenSubtype.POLICE)] * 6
    assert set(_drive(monkeypatch, seq)) == {SirenSubtype.UNKNOWN}


def test_consistent_subtype_is_named_after_enough_votes(monkeypatch):
    """같은 차종이 꾸준히 관측되면 그때는 이름을 붙인다 — 억제이지 봉인이 아니다."""
    out = _drive(monkeypatch, [_siren(SirenSubtype.FIRE)] * 8)
    assert out[0] is SirenSubtype.UNKNOWN                  # 표가 모자란 초반은 보류
    assert out[-1] is SirenSubtype.FIRE
    assert out.index(SirenSubtype.FIRE) >= Pipeline.SUBTYPE_MIN_VOTES - 1


def test_vote_resets_between_events(monkeypatch):
    """앞 사이렌의 표가 다음 사이렌으로 넘어가면 안 된다."""
    seq = ([_siren(SirenSubtype.FIRE)] * 8 + [_NOISE] * 4
           + [_siren(SirenSubtype.AMBULANCE)])
    out = _drive(monkeypatch, seq, hangover=0.0)
    assert out[7] is SirenSubtype.FIRE          # 앞 이벤트는 확정됐고
    assert out[-1] is SirenSubtype.UNKNOWN      # 새 이벤트는 표 1개뿐이라 보류


def test_low_confidence_ticks_abstain_rather_than_vote(monkeypatch):
    """확신 미달 틱은 UNKNOWN 으로 오는데, 그게 표로 세어지면 안 된다(기권)."""
    seq = [_siren(SirenSubtype.UNKNOWN, 0.4)] * 10
    assert set(_drive(monkeypatch, seq)) == {SirenSubtype.UNKNOWN}


def test_reported_confidence_is_model_not_vote_ratio(monkeypatch):
    """표 비율을 모델 확신도 자리에 넣으면 0.62 관측이 1.00 으로 보고된다."""
    import pipeline.runner as R
    it = iter([_siren(SirenSubtype.FIRE, 0.62)] * 8)
    monkeypatch.setattr(R, "classify", lambda chunk: next(it))
    p = Pipeline(clock=lambda: 0.0, emergency_hangover=0.0)
    last = None
    for _ in range(8):
        last = p.process(AudioChunk(samples=np.zeros(SR, np.float32),
                                    sample_rate=SR)).sound
    assert last.subtype is SirenSubtype.FIRE
    assert last.subtype_confidence == pytest.approx(0.62, abs=1e-6)


def test_horn_does_not_carry_a_vehicle_type(monkeypatch):
    out = _drive(monkeypatch, [ClassResult.from_label(SoundClass.HORN, 0.9)])
    assert out[0] is None
