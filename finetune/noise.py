"""도로/차량 소음 풀 준비 (다운로드는 1회, idempotent).

기본 풀 (약 390MB 다운로드):
- DEMAND TCAR/TBUS/STRAFFIC 16kHz zip (Zenodo 1227121, 각 환경 ch01~ch16 × 300s)
  라이선스: CC BY-SA 3.0 으로 취급(레코드 표기 혼재 — 엄격한 쪽).
- MS-SNSD 차량 소음 4파일 (16kHz mono, GitHub raw). 코드 MIT / 오디오 CC0·CC BY-SA 혼재.

사이렌 등 추가 소음: data/noise/extra/ 에 wav 를 넣으면 자동 포함(UrbanSound8K 는
6GB 통짜라 v0 기본 제외 — docs/stt/finetune.md 한계 참고).

실행: python3 -m finetune.noise
"""
from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

NOISE_DIR = Path("data/noise")

_DEMAND = {
    "demand_tcar": "https://zenodo.org/records/1227121/files/TCAR_16k.zip?download=1",
    "demand_tbus": "https://zenodo.org/records/1227121/files/TBUS_16k.zip?download=1",
    "demand_straffic": "https://zenodo.org/records/1227121/files/STRAFFIC_16k.zip?download=1",
}
_MSSNSD = ["Car_1.wav", "Traffic_1.wav", "Bus_1.wav", "Metro_1.wav"]
_MSSNSD_BASE = "https://raw.githubusercontent.com/microsoft/MS-SNSD/master/noise_train/"


def list_noise_files(noise_dir: Path = NOISE_DIR) -> list[Path]:
    """노이즈 풀의 wav 전부(extra/ drop-in 포함), 경로 정렬 — 순서 결정적."""
    return sorted(Path(noise_dir).rglob("*.wav"))


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[noise] 다운로드: {url} → {dest}", file=sys.stderr)
    urllib.request.urlretrieve(url, dest)


def main() -> None:
    for name, url in _DEMAND.items():
        env_dir = NOISE_DIR / name
        if env_dir.exists() and any(env_dir.glob("**/*.wav")):
            print(f"[noise] 스킵(있음): {env_dir}")
            continue
        zip_path = NOISE_DIR / f"{name}.zip"
        _download(url, zip_path)
        with zipfile.ZipFile(zip_path) as zf:       # zip 내부: TCAR/ch01.wav ...
            zf.extractall(env_dir)
        zip_path.unlink()

    ms_dir = NOISE_DIR / "ms_snsd"
    for fname in _MSSNSD:
        dest = ms_dir / fname
        if dest.exists():
            print(f"[noise] 스킵(있음): {dest}")
            continue
        _download(_MSSNSD_BASE + fname, dest)

    (NOISE_DIR / "extra").mkdir(parents=True, exist_ok=True)
    files = list_noise_files()
    print(f"[noise] 풀 준비 완료: wav {len(files)}개 (extra/ 에 사이렌 등 추가 가능)")


if __name__ == "__main__":
    main()
