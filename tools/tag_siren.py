"""사이렌 차종 실시간 태거 — 젯슨에서 녹음하며 눈으로 본 차종을 찍는다.

마이크는 계속 세션 wav 하나에 기록되고, 조수석에서 차종 키를 누르면 그 시점이
tags.csv 에 쌓인다. **사람은 차종만, 사이렌 구간(시간 경계)은 tools/cut_clips.py 가
기존 검출기로 잡는다** — 라벨은 항상 소리보다 늦게 들어오기 때문이다(사이렌이 들리고
→ 차가 보이고 → 키를 누르기까지 10초쯤 지난다).

사용:
  systemctl stop emergency-hud          # HUD 가 ReSpeaker 를 잡고 있으면 먼저 해제
  python tools/tag_siren.py --place 소방서앞              # 젯슨 + ReSpeaker(6ch→ch0)
  python tools/tag_siren.py --place 테스트 --channels 1   # 노트북 내장 마이크

산출:
  data/siren_sessions/20260809_1432_소방서앞/{audio.wav, tags.csv}
  폴더명이 곧 세션 ID이자 장소 기록 — 학습셋을 세션·장소 단위로 나눌 때 쓴다.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.types import SAMPLE_RATE, SirenSubtype

# 키 → 라벨. 값은 core.types.SirenSubtype 를 그대로 써서 학습 쪽에서 재매핑이 없게 한다.
KEYS = {
    "1": SirenSubtype.AMBULANCE,
    "2": SirenSubtype.POLICE,
    "3": SirenSubtype.FIRE,
    "u": SirenSubtype.UNKNOWN,      # 사이렌은 들리는데 차를 못 봄
}
HELP = "1 구급차 · 2 경찰차 · 3 소방차 · u 차종모름 · z 마지막취소 · q 종료"


def _open_session(root: Path, place: str) -> Path:
    """data/siren_sessions/<날짜시각>_<장소>/ 를 만들고 돌려준다."""
    safe = place.replace("/", "-").replace(" ", "") or "unknown"
    d = root / f"{datetime.now().strftime('%Y%m%d_%H%M')}_{safe}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _chunks(channels: int, device: int | None):
    """캡처 소스. ReSpeaker 는 6채널로 열어 ch0(빔포밍 처리본)만 쓴다."""
    from audio.capture import iter_chunks_from_mic, iter_chunks_from_respeaker

    if channels > 1:
        return iter_chunks_from_respeaker(channel=0, num_channels=channels, device=device)
    return iter_chunks_from_mic(channels=1, device=device)


def _record(chunks, wav: wave.Wave_write, state: dict, stop: threading.Event) -> None:
    """캡처 → int16 PCM 으로 세션 wav 에 append. 경과는 쓴 샘플 수로 센다(벽시계 무관)."""
    try:
        for chunk in chunks:
            if stop.is_set():
                break
            x = np.clip(np.asarray(chunk.samples, dtype=np.float32), -1.0, 1.0)
            wav.writeframes((x * 32767.0).astype("<i2").tobytes())
            state["frames"] += x.size
            state["rms"] = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
    except Exception as e:                      # 장치 분리 등 — 메인이 알아야 한다
        state["error"] = e
        stop.set()


def _write_tags(path: Path, tags: list[tuple[float, str]]) -> None:
    """태그 전체를 매번 다시 쓴다 — 수십 줄이라 비용이 없고, 취소가 pop 한 줄로 끝난다."""
    with path.open("w", encoding="utf-8") as f:
        f.write("t_sec,label\n")
        for t, label in tags:
            f.write(f"{t:.1f},{label}\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="사이렌 차종 실시간 태거")
    p.add_argument("--place", required=True, help="녹음 장소 (폴더명에 들어간다) 예: 소방서앞")
    p.add_argument("--channels", type=int, default=6, help="6=ReSpeaker(기본), 1=노트북 마이크")
    p.add_argument("--device", type=int, default=None, help="입력 장치 인덱스 (미지정 시 자동 탐지)")
    p.add_argument("--out", default="data/siren_sessions", help="세션 폴더를 만들 위치")
    a = p.parse_args(argv)

    session = _open_session(Path(a.out), a.place)
    wav_path, tags_path = session / "audio.wav", session / "tags.csv"
    tags: list[tuple[float, str]] = []
    state = {"frames": 0, "rms": 0.0, "error": None}
    stop = threading.Event()

    wav = wave.open(str(wav_path), "wb")
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(SAMPLE_RATE)

    t = threading.Thread(
        target=_record, args=(_chunks(a.channels, a.device), wav, state, stop), daemon=True
    )
    t.start()

    # 첫 청크(또는 캡처 오류)를 기다렸다 시작한다. 장치 오선택·점유(HUD 서비스가 ReSpeaker
    # 를 잡고 있는 경우)를 한 시간 태깅한 뒤가 아니라 지금 드러내야 한다.
    for _ in range(50):
        if state["frames"] or state["error"] is not None:
            break
        time.sleep(0.1)

    print(f"[tag] 세션: {session}")
    if state["error"] is not None:
        print(f"[tag] 캡처 오류: {state['error']}", file=sys.stderr)
        stop.set()
        wav.close()
        return 1
    if not state["frames"]:
        print("[tag] 경고: 5초 동안 입력이 없다 — 장치를 확인할 것 "
              "(HUD 가 마이크를 잡고 있으면 systemctl stop emergency-hud)", file=sys.stderr)
    print(f"[tag] {HELP}")
    _write_tags(tags_path, tags)                # 빈 파일이라도 먼저 만들어 둔다

    try:
        while not stop.is_set():
            key = input("> ").strip().lower()
            if state["error"] is not None:
                break
            if key == "q":
                break
            if key == "z":
                if tags:
                    t_sec, label = tags.pop()
                    _write_tags(tags_path, tags)
                    print(f"[tag] 취소: {t_sec:.1f}s {label}")
                else:
                    print("[tag] 취소할 태그가 없다")
                continue
            if key not in KEYS:
                if key:
                    print(f"[tag] {HELP}")
                continue

            t_sec = state["frames"] / SAMPLE_RATE
            tags.append((t_sec, KEYS[key].value))
            _write_tags(tags_path, tags)
            # dBFS 를 같이 찍어 마이크가 죽어 있으면 태그를 쌓기 전에 알아채게 한다.
            db = 20.0 * np.log10(max(state["rms"], 1e-9))
            print(f"[tag] {t_sec:7.1f}s  {KEYS[key].value:<9} (#{len(tags)}, 입력 {db:.0f} dBFS)")
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        stop.set()
        t.join(timeout=2.0)                     # stream.read 블로킹 때문에 최대 1청크 대기
        wav.close()

    if state["error"] is not None:
        print(f"[tag] 캡처 오류: {state['error']}", file=sys.stderr)

    secs = state["frames"] / SAMPLE_RATE
    print(f"[tag] 종료 — {secs / 60:.1f}분, 태그 {len(tags)}개 → {session}")
    if state["frames"] and state["rms"] == 0.0:
        print("[tag] 경고: 마지막 입력이 무음이다. 장치 선택을 확인할 것.", file=sys.stderr)
    return 1 if state["error"] is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
