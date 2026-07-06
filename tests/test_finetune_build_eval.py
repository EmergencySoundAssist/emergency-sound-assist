"""조건 배분 결정성·manifest 생성 테스트 (tmp 합성 입력, 네트워크 없음)."""
import json

import numpy as np
import soundfile as sf

from finetune.build_eval import assign_conditions, build


def test_assign_corpus_even_split():
    conds = assign_conditions(200, "corpus", seed=1)
    assert len(conds) == 200
    from collections import Counter
    c = Counter(conds)
    assert c[("clean", None)] == 50
    assert c[("noise", 10)] == 50 and c[("noise", 5)] == 50 and c[("noise", 0)] == 50


def test_assign_tts_half_loudspeaker():
    conds = assign_conditions(150, "tts", seed=1)
    n_ls = sum(1 for c, _ in conds if c == "loudspeaker")
    assert n_ls == 75
    assert all(s in (10, 5, 0) for c, s in conds if c == "loudspeaker")


def test_assign_deterministic():
    assert assign_conditions(150, "tts", seed=5) == assign_conditions(150, "tts", seed=5)


def _make_inputs(tmp_path, n_utts=4):
    """가짜 raw_tts/raw_corpus jsonl + wav + 노이즈 1개를 tmp 에 만든다."""
    rng = np.random.default_rng(0)
    (tmp_path / "noise").mkdir()
    sf.write(tmp_path / "noise" / "white.wav",
             (0.05 * rng.standard_normal(16000 * 12)).astype(np.float32), 16000)
    rows = {"raw_tts.jsonl": [], "raw_corpus.jsonl": []}
    for i in range(n_utts):
        for kind, fname in (("tts", "raw_tts.jsonl"), ("corpus", "raw_corpus.jsonl")):
            uid = f"{kind}_{i}"
            wav = tmp_path / f"{uid}.wav"
            tone = (0.1 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)
                    ).astype(np.float32)
            sf.write(wav, tone, 16000)
            row = {"id": uid, "path": str(wav), "source": kind}
            if kind == "tts":
                row.update(text="차 세우세요", category="police",
                           keywords=["세우"], voice="ko-KR-SunHiNeural")
            else:
                row.update(text="일반 문장")
            rows[fname].append(row)
    for fname, rs in rows.items():
        with open(tmp_path / fname, "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return tmp_path


def test_build_writes_manifest_and_wavs(tmp_path):
    root = _make_inputs(tmp_path)
    out = tmp_path / "out"
    n = build(tts_manifest=root / "raw_tts.jsonl",
              corpus_manifest=root / "raw_corpus.jsonl",
              noise_dir=root / "noise", out_dir=out, seed=99)
    assert n == 8
    lines = (out / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 8
    for line in lines:
        row = json.loads(line)
        assert row["condition"] in ("clean", "noise", "loudspeaker")
        assert (out / "wav" / f"{row['id']}.wav").exists()
        assert row["ref_text"]


def test_build_deterministic(tmp_path):
    root = _make_inputs(tmp_path)
    out1, out2 = tmp_path / "o1", tmp_path / "o2"
    build(tts_manifest=root / "raw_tts.jsonl", corpus_manifest=root / "raw_corpus.jsonl",
          noise_dir=root / "noise", out_dir=out1, seed=99)
    build(tts_manifest=root / "raw_tts.jsonl", corpus_manifest=root / "raw_corpus.jsonl",
          noise_dir=root / "noise", out_dir=out2, seed=99)
    def _rows(p):
        rows = [json.loads(l) for l in
                (p / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
        for r in rows:
            r.pop("path")                    # 출력 디렉토리 경로만 다르고 나머지 동일
        return rows
    assert _rows(out1) == _rows(out2)
