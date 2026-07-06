"""TTS 잡 계획 테스트 — 실제 합성(네트워크)은 여기서 안 한다."""
from finetune.phrases import load_phrases
from finetune.tts import plan_jobs, EVAL_VOICES, TRAIN_VOICES


def test_plan_jobs_count_and_ids():
    rows = load_phrases("finetune/emergency_phrases.csv")
    jobs = plan_jobs(rows, EVAL_VOICES)
    assert len(jobs) == len(rows) * 2          # 75문장 × 2화자 = 150
    ids = [j["id"] for j in jobs]
    assert len(set(ids)) == len(ids)           # 유일
    assert jobs[0]["id"] == "tts_p000_sunhi" and jobs[1]["id"] == "tts_p000_injoon"


def test_plan_jobs_deterministic():
    rows = load_phrases("finetune/emergency_phrases.csv")
    assert plan_jobs(rows, EVAL_VOICES) == plan_jobs(rows, EVAL_VOICES)


def test_eval_and_train_voices_disjoint():
    assert not set(EVAL_VOICES) & set(TRAIN_VOICES)
