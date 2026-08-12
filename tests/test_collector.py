"""SirenCollector — 자동/수동 트리거, 프리롤, grace 라벨, csv 왕복.

시간축이 '흘러간 오디오'라서 벽시계 mocking 없이 on_result 를 밀어넣는 것만으로
전 시나리오가 결정적으로 재현된다.
"""

import csv
import wave

import numpy as np
import pytest

from collect.collector import LABEL_UNLABELED, SirenCollector
from core.types import AudioChunk, ClassResult, SirenSubtype, SoundClass

SR = 16_000


def _chunk(sec=1.0, value=0.1):
    return AudioChunk(samples=np.full(int(SR * sec), value, dtype=np.float32), sample_rate=SR)


def _siren(conf=0.9, subtype=None, sub_conf=None):
    return ClassResult.from_label(SoundClass.SIREN, conf, subtype=subtype,
                                  subtype_confidence=sub_conf)


def _noise():
    return ClassResult.from_label(SoundClass.NORMAL_TRAFFIC, 0.8)


def _horn():
    return ClassResult.from_label(SoundClass.HORN, 0.9)


def _mk(tmp_path, **kw):
    kw.setdefault("pre_roll", 3.0)
    kw.setdefault("auto_tail", 2.0)
    kw.setdefault("manual_sec", 4.0)
    kw.setdefault("max_sec", 30.0)
    kw.setdefault("grace", 5.0)
    return SirenCollector(out_root=tmp_path, place="테스트", **kw)


def _rows(c):
    with (c.session / "labels.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _feed(c, n, cls_fn, value=0.1):
    for _ in range(n):
        c.on_result(_chunk(value=value), cls_fn())


def _wav_sec(path):
    with wave.open(str(path), "rb") as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, SR)
        return w.getnframes() / SR


# ── 자동 트리거 ─────────────────────────────────────────────────────────

def test_auto_clip_includes_preroll_and_tail(tmp_path):
    c = _mk(tmp_path)
    _feed(c, 5, _noise)                 # 링버퍼는 pre_roll(3초)만 유지해야 한다
    _feed(c, 4, _siren)                 # 감지 → 자동 녹음
    _feed(c, 2, _noise)                 # auto_tail(2초) 지나면 닫힘
    rows = _rows(c)
    assert len(rows) == 1
    r = rows[0]
    assert r["trigger"] == "auto"
    assert r["label"] == LABEL_UNLABELED
    assert float(r["pre_roll_sec"]) == pytest.approx(3.0)
    # 프리롤 3(트리거 청크 제외한 직전 3청크) + 사이렌 4 + 꼬리 2 = 9초
    assert float(r["duration_sec"]) == pytest.approx(9.0)
    assert _wav_sec(c.session / r["clip"]) == pytest.approx(9.0)
    assert r["det_flags"] == "111100"   # 트리거 틱부터: 사이렌 4 + 꼬리 2


def test_horn_also_triggers_and_is_marked_apart(tmp_path):
    """경적도 긴급으로 HUD 를 띄우므로 같이 모은다 — 안 모으면 경적 쪽 지연·오검출은
    데이터가 없어 분석 자체가 불가능하다(1차 수집에서 실제로 그랬다).
    다만 무엇이 클립을 열었는지는 trigger_class 로 갈라 둔다."""
    c = _mk(tmp_path)
    _feed(c, 3, _horn)
    assert c.status()["recording"] == "auto"
    _feed(c, 2, _noise)
    r = _rows(c)[0]
    assert r["trigger_class"] == "horn"
    assert r["det_flags"] == "00000"        # 사이렌 판정은 한 번도 없었다
    assert r["horn_flags"] == "11100"       # 경적은 트리거 틱부터 3틱
    assert r["label"] == LABEL_UNLABELED    # 라벨은 사람이 붙인다


def test_siren_trigger_is_marked_as_siren(tmp_path):
    c = _mk(tmp_path)
    _feed(c, 2, _siren)
    _feed(c, 2, _noise)
    r = _rows(c)[0]
    assert r["trigger_class"] == "siren"
    assert r["det_flags"] == "1100"


def test_horn_label_never_opens_a_manual_clip(tmp_path):
    """'경적이었다'는 도장이지 새로 녹음할 소리가 아니다."""
    c = _mk(tmp_path)
    _feed(c, 3, _noise)
    c.on_label("horn")
    assert c.status()["recording"] is None
    assert _rows(c) == []


def test_horn_label_stamps_the_previous_clip(tmp_path):
    c = _mk(tmp_path)
    _feed(c, 3, _horn)
    _feed(c, 2, _noise)                 # unlabeled 로 닫힘
    c.on_label("horn")
    assert _rows(c)[0]["label"] == "horn"


def test_debounce_gap_does_not_split_clip(tmp_path):
    """auto_tail 안에서 판정이 잠깐 끊겨도 클립은 하나로 이어진다."""
    c = _mk(tmp_path)
    _feed(c, 3, _siren)
    _feed(c, 1, _noise)                 # 1초 < auto_tail 2초
    _feed(c, 3, _siren)
    _feed(c, 2, _noise)
    assert len(_rows(c)) == 1
    assert float(_rows(c)[0]["duration_sec"]) == pytest.approx(9.0)


def test_max_sec_caps_runaway_clip(tmp_path):
    c = _mk(tmp_path, max_sec=6.0)
    _feed(c, 10, _siren)
    rows = _rows(c)
    assert float(rows[0]["duration_sec"]) == pytest.approx(6.0)
    # 상한으로 닫힌 뒤에도 사이렌이 이어지면 다음 틱에서 새 자동 클립이 열린다
    assert c.status()["recording"] == "auto"


def test_model_prediction_is_recorded_for_comparison(tmp_path):
    c = _mk(tmp_path)
    c.on_result(_chunk(), _siren(conf=0.7, subtype=SirenSubtype.POLICE, sub_conf=0.5))
    c.on_result(_chunk(), _siren(conf=0.95, subtype=SirenSubtype.AMBULANCE, sub_conf=0.8))
    _feed(c, 2, _noise)
    r = _rows(c)[0]
    assert float(r["det_conf_max"]) == pytest.approx(0.95)
    assert r["model_subtype"] == "ambulance"        # 확신 높은 쪽이 남는다
    assert float(r["model_sub_conf"]) == pytest.approx(0.8)


# ── 라벨 버튼 ───────────────────────────────────────────────────────────

def test_label_during_recording(tmp_path):
    c = _mk(tmp_path)
    _feed(c, 2, _siren)
    c.on_label(SirenSubtype.AMBULANCE)
    _feed(c, 2, _noise)
    assert _rows(c)[0]["label"] == "ambulance"


def test_label_within_grace_lands_on_last_clip(tmp_path):
    """라벨은 소리보다 늦다 — 닫힌 직후의 키는 직전 클립을 가리켜야 한다."""
    c = _mk(tmp_path)
    _feed(c, 2, _siren)
    _feed(c, 2, _noise)                 # 클립 닫힘 (unlabeled)
    _feed(c, 3, _noise)                 # grace(5초) 안
    c.on_label(SirenSubtype.FIRE)
    assert _rows(c)[0]["label"] == "fire"
    assert c.status()["recording"] is None      # 수동 녹음이 열리면 안 된다


def test_label_after_grace_opens_manual_clip(tmp_path):
    c = _mk(tmp_path)
    _feed(c, 2, _siren)
    _feed(c, 2, _noise)
    _feed(c, 6, _noise)                 # grace(5초) 지남
    c.on_label(SirenSubtype.POLICE)     # → 미검출 표본 수동 녹음
    assert c.status()["recording"] == "manual"
    assert _rows(c)[0]["label"] == LABEL_UNLABELED      # 직전 클립은 그대로


def test_not_siren_stamps_false_positive_and_clears_the_queue(tmp_path):
    """오검출 클립에 사이렌아님 도장 → 이후 라벨이 그 클립에 흡수되지 않는다."""
    c = _mk(tmp_path)
    _feed(c, 2, _siren)                 # 오검출(FP) 자동 클립
    _feed(c, 2, _noise)                 # unlabeled 로 닫힘
    c.on_label("not_siren")
    assert _rows(c)[0]["label"] == "not_siren"
    c.on_label(SirenSubtype.FIRE)       # 다음 라벨은 FP 클립을 건너뛴다
    assert _rows(c)[0]["label"] == "not_siren"
    assert c.status()["recording"] == "manual"      # → 미검출 수동 녹음으로


def test_not_siren_never_opens_a_manual_clip(tmp_path):
    """'사이렌이 아니다'는 새로 녹음할 소리가 없다 — 대상 없으면 도장만 거부."""
    c = _mk(tmp_path)
    _feed(c, 3, _noise)
    c.on_label("not_siren")
    assert c.status()["recording"] is None
    assert _rows(c) == []


def test_label_never_overwrites_a_labeled_clip(tmp_path):
    """grace 안이라도 라벨이 이미 있으면 새 키는 수동 녹음이다 — 데이터를 덮지 않는다."""
    c = _mk(tmp_path)
    _feed(c, 2, _siren)
    c.on_label(SirenSubtype.AMBULANCE)
    _feed(c, 2, _noise)                 # ambulance 로 닫힘
    c.on_label(SirenSubtype.FIRE)       # grace 안이지만 → 수동 녹음
    assert c.status()["recording"] == "manual"
    assert _rows(c)[0]["label"] == "ambulance"


def test_cancel_then_relabel_fixes_a_mistake(tmp_path):
    c = _mk(tmp_path)
    _feed(c, 2, _siren)
    c.on_label(SirenSubtype.AMBULANCE)
    _feed(c, 2, _noise)
    c.on_cancel()                       # 직전 클립 라벨 취소
    assert _rows(c)[0]["label"] == LABEL_UNLABELED
    c.on_label(SirenSubtype.POLICE)     # unlabeled 가 됐으니 grace 라벨이 붙는다
    assert _rows(c)[0]["label"] == "police"


def test_late_label_lands_on_previous_vehicle_not_the_open_clip(tmp_path):
    """연속 출동: 앞차 클립이 미라벨로 닫히고 뒷차 클립이 이미 열린 뒤 도착한
    라벨은 앞차 몫이다 — 라벨은 항상 소리보다 ~10초 늦게 오기 때문."""
    c = _mk(tmp_path)
    _feed(c, 3, _siren)                 # 차량 A
    _feed(c, 2, _noise)                 # A 닫힘 (unlabeled)
    _feed(c, 2, _noise)
    _feed(c, 2, _siren)                 # 차량 B → 새 자동 클립
    assert c.status()["recording"] == "auto"
    c.on_label(SirenSubtype.AMBULANCE)  # A 를 보고 누른 라벨
    assert _rows(c)[0]["label"] == "ambulance"          # A 에 붙는다
    assert c.status()["label"] == LABEL_UNLABELED       # B 는 그대로
    c.on_label(SirenSubtype.FIRE)       # 이제 B 차례
    assert c.status()["label"] == "fire"
    _feed(c, 2, _noise)
    assert _rows(c)[1]["label"] == "fire"


def test_presses_column_records_mid_recording_labels(tmp_path):
    """병합 클립(연속 차량)을 오프라인에서 가르는 근거 — 누름 시각:라벨 기록."""
    c = _mk(tmp_path)
    _feed(c, 2, _siren)
    c.on_label(SirenSubtype.AMBULANCE)
    _feed(c, 3, _siren)
    c.on_label(SirenSubtype.FIRE)
    _feed(c, 2, _noise)
    r = _rows(c)[0]
    assert r["label"] == "fire"                          # 마지막 누름이 최종 라벨
    assert r["presses"] == "2.0:ambulance|5.0:fire"


def test_cancel_during_recording_pops_last_press(tmp_path):
    c = _mk(tmp_path)
    _feed(c, 2, _siren)
    c.on_label(SirenSubtype.AMBULANCE)
    _feed(c, 1, _siren)
    c.on_label(SirenSubtype.FIRE)
    c.on_cancel()                       # fire 취소 → ambulance 복귀
    assert c.status()["label"] == "ambulance"
    _feed(c, 2, _noise)
    assert _rows(c)[0]["label"] == "ambulance"
    assert _rows(c)[0]["presses"] == "2.0:ambulance"


def test_inputs_after_close_are_ignored(tmp_path):
    """close() 뒤(파이프라인 종료·HUD 만 생존) 버튼은 유령 클립을 열면 안 된다."""
    c = _mk(tmp_path)
    _feed(c, 2, _siren)
    assert c.close() is not None
    assert c.close() is None            # 멱등 — 두 번째 호출은 조용히 무시
    c.on_label(SirenSubtype.FIRE)
    st = c.status()
    assert st["recording"] is None and st["closed"] is True
    assert _rows(c)[0]["label"] == LABEL_UNLABELED
    c.on_result(_chunk(), _siren())
    assert c.status()["clips"] == 1


def test_session_dirs_never_collide(tmp_path):
    """같은 초에 재시작해도 세션 폴더가 겹치지 않는다 — 겹치면 직전 세션이 지워진다."""
    dirs = [_mk(tmp_path).session for _ in range(3)]
    assert len(set(dirs)) == 3
    assert all(d.exists() for d in dirs)


# ── 수동(미검출) 트리거 ─────────────────────────────────────────────────

def test_manual_clip_records_preroll_plus_target(tmp_path):
    c = _mk(tmp_path)
    _feed(c, 5, _noise)                 # 링버퍼 채움
    c.on_label(SirenSubtype.FIRE)       # 미검출 사이렌 신고
    _feed(c, 4, _noise)                 # manual_sec(4초) 채우면 닫힘
    rows = _rows(c)
    assert len(rows) == 1
    r = rows[0]
    assert (r["trigger"], r["label"]) == ("manual", "fire")
    # 수동은 버튼 이전 과거가 전부 프리롤이다: 링이 든 pre_roll+1청크 = 4초.
    # (자동은 마지막 청크가 '트리거 틱'이라 프리롤이 정확히 pre_roll=3초가 된다.)
    assert float(r["pre_roll_sec"]) == pytest.approx(4.0)
    assert float(r["duration_sec"]) == pytest.approx(4.0 + 4.0)
    assert _wav_sec(c.session / r["clip"]) == pytest.approx(8.0)


def test_manual_clip_extends_while_detector_fires(tmp_path):
    """수동 마감 시점에 감지가 살아 있으면 사이렌이 끝날 때까지 이어 담는다."""
    c = _mk(tmp_path)
    c.on_label(SirenSubtype.POLICE)
    _feed(c, 3, _noise)
    _feed(c, 4, _siren)                 # 4초째(마감)에도 사이렌 → 연장
    assert c.status()["recording"] == "manual"
    _feed(c, 1, _noise)                 # 판정 끊기면 닫힌다
    assert float(_rows(c)[0]["duration_sec"]) == pytest.approx(8.0)


# ── 링버퍼·종료 ─────────────────────────────────────────────────────────

def test_ring_buffer_holds_only_preroll(tmp_path):
    c = _mk(tmp_path, pre_roll=2.0)
    _feed(c, 30, _noise)
    _feed(c, 1, _siren)
    c2 = c.status()
    assert c2["recording"] == "auto"
    _feed(c, 2, _noise)
    assert float(_rows(c)[0]["pre_roll_sec"]) == pytest.approx(2.0)


def test_close_saves_open_clip_and_summarises(tmp_path):
    c = _mk(tmp_path)
    _feed(c, 3, _siren)
    c.on_label(SirenSubtype.AMBULANCE)
    summary = c.close()                 # 녹음 중 종료 → 저장돼야 한다
    assert len(_rows(c)) == 1
    assert _rows(c)[0]["label"] == "ambulance"
    assert "ambulance 1" in summary


def test_status_counts_and_feedback(tmp_path):
    t = {"v": 0.0}
    c = _mk(tmp_path, clock=lambda: t["v"])
    _feed(c, 2, _siren)
    _feed(c, 2, _noise)
    c.on_label(SirenSubtype.FIRE)
    st = c.status()
    assert st["counts"] == {"fire": 1}
    assert "소방차" in st["feedback"]
    t["v"] = 10.0                       # FEEDBACK_SECONDS(4초) 지나면 사라진다
    assert c.status()["feedback"] is None


# ── HUD 레이아웃 계약 ───────────────────────────────────────────────────

def test_collect_button_rects_fit_and_do_not_overlap():
    from hud.card import collect_button_rects

    for w, h in ((1280, 360), (800, 480), (320, 180)):
        rects = collect_button_rects(w, h)
        assert len(rects) == 6
        for x, y, bw, bh in rects:
            assert 0 <= x and x + bw <= w
            assert 0 <= y and y + bh <= h
            assert bh >= 34                 # 터치 가능한 최소 높이
        xs = sorted(rects, key=lambda r: r[0])
        for a, b in zip(xs, xs[1:]):
            assert a[0] + a[2] <= b[0]      # 서로 겹치지 않는다


def test_horn_clip_is_not_exported_as_a_siren_negative(tmp_path):
    """경적 클립은 det_flags 가 전부 0이라 네거티브 스크린을 그냥 통과한다.
    그대로 두면 경적 오디오가 noise 로 학습돼 경적 검출이 망가진다."""
    import sys, types
    from tools.export_hardneg import _screen
    row = {"trigger": "auto", "trigger_class": "horn",
           "det_flags": "0000", "det_conf_max": "0.0"}
    ok, why = _screen(row, np.zeros(16_000, dtype=np.float32))
    assert not ok and "경적" in why


def test_siren_clip_still_screens_normally():
    from tools.export_hardneg import _screen
    row = {"trigger": "auto", "trigger_class": "siren",
           "det_flags": "1000", "det_conf_max": "0.5"}
    ok, _ = _screen(row, np.zeros(16_000, dtype=np.float32))
    assert ok            # 약한 단발 검출 + 무음 → 네거티브로 통과
