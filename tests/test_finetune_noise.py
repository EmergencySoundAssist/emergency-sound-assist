"""노이즈 풀 스캔·코퍼스 샘플링의 결정성 테스트 (다운로드 없음)."""
import numpy as np
import soundfile as sf

from finetune.corpora import select_indices
from finetune.noise import list_noise_files


def test_select_indices_deterministic_sorted_unique():
    a = select_indices(457, 130, seed=20260706)
    b = select_indices(457, 130, seed=20260706)
    assert a == b and a == sorted(a) and len(set(a)) == 130
    assert all(0 <= i < 457 for i in a)


def test_select_indices_caps_at_total():
    assert len(select_indices(50, 130, seed=1)) == 50


def test_list_noise_files_recursive_and_sorted(tmp_path):
    (tmp_path / "demand_tcar").mkdir()
    (tmp_path / "extra").mkdir()
    x = (0.1 * np.random.default_rng(0).standard_normal(16000)).astype(np.float32)
    sf.write(tmp_path / "demand_tcar" / "ch01.wav", x, 16000)
    sf.write(tmp_path / "extra" / "siren.wav", x, 16000)
    (tmp_path / "readme.txt").write_text("not audio")
    files = list_noise_files(tmp_path)
    assert [f.name for f in files] == ["ch01.wav", "siren.wav"]
