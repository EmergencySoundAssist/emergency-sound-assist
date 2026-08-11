"""cut_collect 순수 로직 — fixed 폴백 상한, presses 파싱, 병합 클립 구간 라벨 배정,
오검출 창 추출."""

import numpy as np

import tools.cut_collect as cc
from tools.cut_collect import _assign_spans, _fixed_span, _negative_windows, _parse_presses


def test_fixed_span_is_capped():
    """오누름·사이렌 종료 후의 소음이 클립 끝까지 라벨을 달고 가면 안 된다."""
    assert _fixed_span(60.0, 10.0, 2.5) == (7.5, 22.5)      # cap 15초
    assert _fixed_span(10.0, 4.0, 2.5) == (1.5, 10.0)       # 클립이 짧으면 끝까지
    assert _fixed_span(30.0, 1.0, 2.5) == (0.0, 15.0)       # 앞이 모자라면 0부터


def test_parse_presses_roundtrip():
    assert _parse_presses("") == []
    assert _parse_presses("12.0:ambulance|30.5:fire") == [
        (12.0, "ambulance"), (30.5, "fire")]


def test_assign_spans_single_label_covers_all():
    spans = [(2.0, 10.0), (20.0, 30.0)]
    assert _assign_spans(spans, [(12.0, "fire")], "fire") == [
        (2.0, 10.0, "fire"), (20.0, 30.0, "fire")]
    assert _assign_spans(spans, [], "police") == [
        (2.0, 10.0, "police"), (20.0, 30.0, "police")]


def test_assign_spans_merged_vehicles_get_their_own_press():
    """구급차(2~20s)+소방차(24~40s)가 auto_tail 안에 이어져 한 클립이 된 경우."""
    spans = [(2.0, 20.0), (24.0, 40.0)]
    presses = [(12.0, "ambulance"), (30.0, "fire")]
    assert _assign_spans(spans, presses, "fire") == [
        (2.0, 20.0, "ambulance"), (24.0, 40.0, "fire")]


def _flags(monkeypatch, flags):
    """검출기 판정열을 주입한다 — ONNX 추론 없이 창 계산만 검증."""
    monkeypatch.setattr(cc, "_siren_flags", lambda audio: flags)
    return np.zeros(16_000 * len(flags), dtype=np.float32)


def test_negative_windows_extract_the_window_that_fooled_the_detector(monkeypatch):
    """오검출은 대부분 단발 발화 — 사이렌 구간 복원 로직으로는 0개가 나온다.
    검출기가 그 tick 에 실제로 본 5초 창 [(i+1)-WIN, i+1) 을 그대로 오려야 한다."""
    audio = _flags(monkeypatch, [False] * 9 + [True] + [False] * 3)   # tick 9 단발
    assert _negative_windows(audio) == [5.0]                           # [5,10)


def test_negative_windows_skip_clip_head_and_overlaps(monkeypatch):
    # tick 2 는 창이 음수로 시작(온전한 5초 없음) → 버린다
    audio = _flags(monkeypatch, [False, False, True] + [False] * 5)
    assert _negative_windows(audio) == []
    # 연속 발화는 같은 소리 — 겹치지 않는 창만 남긴다.
    # tick 4~11 발화 → 창 후보는 0,1,…,7 초지만 겹치지 않는 건 0 과 5 뿐이다.
    audio = _flags(monkeypatch, [False] * 4 + [True] * 8 + [False] * 2)
    assert _negative_windows(audio) == [0.0, 5.0]


def test_assign_spans_unmatched_span_is_dropped():
    """어느 차인지 모르는 구간은 버린다 — 잘못 붙이느니 버린다(cut_clips 원칙)."""
    spans = [(2.0, 20.0), (24.0, 40.0)]
    presses = [(12.0, "ambulance"), (13.0, "fire")]     # 둘 다 첫 구간 근처
    out = _assign_spans(spans, presses, "x")
    assert out[0] == (2.0, 20.0, "ambulance")
    assert out[1] == (24.0, 40.0, None)                 # 25초 이후 누름 없음
