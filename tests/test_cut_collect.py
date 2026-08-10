"""cut_collect 순수 로직 — fixed 폴백 상한, presses 파싱, 병합 클립 구간 라벨 배정."""

from tools.cut_collect import _assign_spans, _fixed_span, _parse_presses


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


def test_assign_spans_unmatched_span_is_dropped():
    """어느 차인지 모르는 구간은 버린다 — 잘못 붙이느니 버린다(cut_clips 원칙)."""
    spans = [(2.0, 20.0), (24.0, 40.0)]
    presses = [(12.0, "ambulance"), (13.0, "fire")]     # 둘 다 첫 구간 근처
    out = _assign_spans(spans, presses, "x")
    assert out[0] == (2.0, 20.0, "ambulance")
    assert out[1] == (24.0, 40.0, None)                 # 25초 이후 누름 없음
