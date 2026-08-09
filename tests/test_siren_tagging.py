"""태거 ↔ 클립 절단의 순수 로직 — 구간 검출·창 분할·겹침 배제·tags.csv 왕복."""

import numpy as np

from tools.cut_clips import _overlaps, _read_tags, _runs, _windows, _read_wav, _write_wav
from tools.tag_siren import _write_tags


def test_runs_finds_contiguous_true_spans():
    assert _runs([]) == []
    assert _runs([False, False]) == []
    assert _runs([True, True, False, True]) == [(0, 2), (3, 4)]
    assert _runs([True, True, True]) == [(0, 3)]        # 끝까지 True 면 닫아준다


def test_windows_drops_the_tail_shorter_than_a_full_clip():
    assert _windows(0.0, 12.0) == [0.0, 5.0]           # 남는 2초는 버린다
    assert _windows(0.0, 10.0) == [0.0, 5.0]           # 경계는 포함
    assert _windows(3.0, 7.0) == []                     # 5초가 안 되면 클립 없음


def test_overlapping_clips_are_rejected():
    """소방서 앞에선 태그 창이 겹친다 — 같은 오디오가 두 라벨로 저장되면 안 된다."""
    taken = [(10.0, 15.0)]
    assert _overlaps(12.0, taken)                       # 안쪽
    assert _overlaps(6.0, taken)                        # 앞에서 걸침
    assert not _overlaps(15.0, taken)                   # 딱 붙는 건 겹침 아님
    assert not _overlaps(5.0, taken)


def test_tags_written_by_tagger_are_read_back_by_cutter(tmp_path):
    """두 도구의 유일한 계약. 취소(pop 후 재작성)도 그대로 반영돼야 한다."""
    path = tmp_path / "tags.csv"
    tags = [(12.3, "ambulance"), (98.7, "fire")]
    _write_tags(path, tags)
    assert _read_tags(path) == [(12.3, "ambulance"), (98.7, "fire")]

    tags.pop()
    _write_tags(path, tags)
    assert _read_tags(path) == [(12.3, "ambulance")]

    _write_tags(path, [])
    assert _read_tags(path) == []


def test_wav_roundtrip_keeps_samples_within_quantisation_error(tmp_path):
    x = np.sin(np.linspace(0, 40, 16000, dtype=np.float32))
    path = tmp_path / "audio.wav"
    _write_wav(path, x)
    assert np.abs(_read_wav(path) - x).max() < 1e-4
