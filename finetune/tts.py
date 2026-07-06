"""긴급문구 TTS 합성 (edge-tts, 네트워크 필요).

- 평가용 화자: SunHi(여)·InJoon(남) — edge-tts 의 한국어 네이티브 보이스는 이 둘 +
  HyunsuMultilingual(남) 뿐이다(2026-07 확인). Hyunsu 는 미래 학습셋용으로 남겨
  화자 누수를 피한다. 화자 풀이 좁은 건 알려진 한계 → docs/stt/finetune.md.
- 출력 포맷은 mp3(24kHz) 고정(엔드포인트 제약) → soundfile 로 읽어 16k wav 저장.
- 비공식 엔드포인트라 일시 403 가능 → 재시도 + 순차 호출(병렬 금지).

실행: python3 -m finetune.tts        # data/eval_v0/tts/*.wav + raw_tts.jsonl
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

from .audio_io import load_mono_16k, save_wav_16k
from .phrases import load_phrases

EVAL_VOICES = ["ko-KR-SunHiNeural", "ko-KR-InJoonNeural"]
TRAIN_VOICES = ["ko-KR-HyunsuMultilingualNeural"]   # 평가에 쓰지 말 것(화자 분리)

_SHORT = {"ko-KR-SunHiNeural": "sunhi", "ko-KR-InJoonNeural": "injoon",
          "ko-KR-HyunsuMultilingualNeural": "hyunsu"}
# 화자별 살짝 다른 프로소디 → 같은 문장이라도 변화를 준다
_RATE = {"ko-KR-SunHiNeural": "+0%", "ko-KR-InJoonNeural": "-5%",
         "ko-KR-HyunsuMultilingualNeural": "+5%"}

PHRASES_CSV = Path("finetune/emergency_phrases.csv")
OUT_DIR = Path("data/eval_v0/tts")
MANIFEST = Path("data/eval_v0/raw_tts.jsonl")


def plan_jobs(rows: list[dict], voices: list[str]) -> list[dict]:
    """문장×화자 잡 목록(결정적 — 파일명·순서가 항상 같다)."""
    jobs = []
    for i, r in enumerate(rows):
        for v in voices:
            jobs.append({
                "id": f"tts_p{i:03d}_{_SHORT[v]}",
                "text": r["text"], "category": r["category"],
                "keywords": r["keywords"], "voice": v,
                "rate": _RATE[v], "pitch": "+0Hz",
            })
    return jobs


def _synth_mp3(text: str, voice: str, rate: str, pitch: str, mp3_path: str,
               retries: int = 4) -> None:
    import edge_tts
    for attempt in range(retries):
        try:
            edge_tts.Communicate(text, voice, rate=rate, pitch=pitch).save_sync(mp3_path)
            return
        except Exception as e:                       # noqa: BLE001 — 백오프 재시도
            if attempt == retries - 1:
                raise
            wait = 2.0 ** attempt
            print(f"[tts] 재시도 {attempt+1}/{retries} ({e}) — {wait}s 대기",
                  file=sys.stderr)
            time.sleep(wait)


def main() -> None:
    rows = load_phrases(PHRASES_CSV)
    jobs = plan_jobs(rows, EVAL_VOICES)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    entries = []
    for n, job in enumerate(jobs, 1):
        wav_path = OUT_DIR / f"{job['id']}.wav"
        if not wav_path.exists():                    # 재실행 시 스킵(재개 가능)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                _synth_mp3(job["text"], job["voice"], job["rate"], job["pitch"],
                           tmp.name)
                save_wav_16k(wav_path, load_mono_16k(tmp.name))
                Path(tmp.name).unlink(missing_ok=True)
            time.sleep(0.3)                          # 엔드포인트 예의(순차+간격)
            print(f"[tts] {n}/{len(jobs)} {job['id']} \"{job['text']}\"")
        entries.append({**job, "path": str(wav_path), "source": "tts"})

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"[tts] 완료: {len(entries)}건 → {MANIFEST}")


if __name__ == "__main__":
    main()
