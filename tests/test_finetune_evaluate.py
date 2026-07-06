"""평가 하네스 테스트 — 가짜 인식기 주입, CER 계산·리포트 검증 (모델 로드 없음)."""
import json

import numpy as np
import soundfile as sf

from finetune.evaluate import aggregate, make_bias_kwargs, run_eval
from finetune.phrases import load_phrases


def test_aggregate_exact_cer():
    rows = [
        # ref 정규화 "차세우세요"(5자), hyp "차세우세요" → CER 0
        {"ref": "차 세우세요", "hyp": "차 세우세요.", "condition": "clean",
         "snr_db": None, "source": "tts", "keywords": ["세우"]},
        # ref "구급차지나갑니다"(8자), hyp "구급차지나갑니다"에서 1자 치환 → CER 1/8
        {"ref": "구급차 지나갑니다", "hyp": "구급차 지나감니다", "condition": "noise",
         "snr_db": 5, "source": "tts", "keywords": ["구급차"]},
    ]
    agg = aggregate(rows)
    assert agg["overall"]["n"] == 2
    assert abs(agg["overall"]["cer"] - (1 / 13)) < 1e-6   # (0+1)/(5+8) corpus-level
    assert abs(agg["by_condition"]["noise@5"]["cer"] - 1 / 8) < 1e-6
    assert abs(agg["by_condition_source"]["tts/noise@5"]["cer"] - 1 / 8) < 1e-6
    assert agg["keyword_hit_rate"] == 1.0 and agg["keyword_n"] == 2
    assert agg["keyword_hit_by_condition"]["clean"] == 1.0


def test_aggregate_keyword_miss():
    rows = [{"ref": "음주 단속 중입니다", "hyp": "음주 운전입니다",
             "condition": "clean", "snr_db": None, "source": "tts",
             "keywords": ["음주", "단속"]}]
    assert aggregate(rows)["keyword_hit_rate"] == 0.0


def test_aggregate_no_keywords_group():
    rows = [{"ref": "일반 문장", "hyp": "일반 문장", "condition": "clean",
             "snr_db": None, "source": "zeroth", "keywords": None}]
    agg = aggregate(rows)
    assert agg["keyword_hit_rate"] is None and agg["keyword_n"] == 0
    assert agg["by_source"]["zeroth"]["cer"] == 0.0
    assert agg["by_condition_source"]["corpus/clean"]["cer"] == 0.0   # 게이트용 그룹


def test_make_bias_kwargs():
    kw = make_bias_kwargs(load_phrases("finetune/emergency_phrases.csv"))
    assert "initial_prompt" in kw and "hotwords" in kw
    assert "구급차" in kw["hotwords"] and len(kw["hotwords"]) < 500
    assert "살려" in kw["hotwords"]      # 마지막 카테고리(shout)가 잘리지 않았는지


def test_run_eval_with_fake_transcriber(tmp_path):
    wav = tmp_path / "a.wav"
    sf.write(wav, np.zeros(16000, dtype=np.float32), 16000)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({
        "id": "x", "path": str(wav), "ref_text": "차 세우세요", "source": "tts",
        "condition": "clean", "snr_db": None, "category": "police",
        "keywords": ["세우"], "tts_voice": "v",
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    report = run_eval(manifest_path=manifest, out_dir=tmp_path / "reports",
                      arm_name="fake", transcribe_fn=lambda path: "차 세우세요")
    assert report["overall"]["cer"] == 0.0
    assert (tmp_path / "reports" / "fake.json").exists()
    assert (tmp_path / "reports" / "fake.md").exists()
