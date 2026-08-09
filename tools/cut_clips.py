"""태깅 세션 → 차종 재학습용 5초 클립.

tools/tag_siren.py 가 남긴 (audio.wav, tags.csv)를 읽어, 태그 시점 **앞쪽으로 되감은**
창 안에서 기존 사이렌 검출기로 실제 사이렌 구간을 확정하고 5초씩 자른다.
되감는 이유: 사람이 차를 눈으로 보고 키를 누를 때쯤이면 사이렌은 이미 한참 울리고 있다.

젯슨 말고 노트북에서 돌리면 된다(파일만 옮기면 됨).

사용:
  python tools/cut_clips.py                     # data/siren_sessions/* → data/siren_clips/
  python tools/cut_clips.py --pre 30 --post 15  # 되감기·앞보기 창 조정

산출:
  data/siren_clips/{ambulance,police,fire,unknown}/<세션>_<번호>.wav
  data/siren_clips/labels.csv   (clip,label,session_id,place,t_start_sec)

labels.csv 는 매 실행마다 **처리한 세션 기준으로 새로 쓴다.**
train/test 는 반드시 session_id 또는 place 단위로 나눌 것 — 같은 출동에서 잘린 클립들은
사실상 같은 소리라, 섞어 나누면 정확도만 부풀고 실차에서 무너진다.
"""
from __future__ import annotations

import argparse
import os
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.types import SAMPLE_RATE, AudioChunk

WIN = 5.0          # 클립 길이 = 차종 모델 입력 창(5초)과 동일
TICK = 1.0         # analyze 한 번이 먹는 길이
GAP = 2            # 이 tick 이하의 판정 구멍은 메운다 (실제 녹음은 경계에서 1~2초 튄다)


def _runs(flags: list[bool]) -> list[tuple[int, int]]:
    """연속 True 구간을 (start, end) 로. end 는 배타.

    [T,T,F,T] → [(0,2),(3,4)]
    """
    out: list[tuple[int, int]] = []
    start = None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(flags)))
    return out


def _merge(runs: list[tuple[int, int]], n_flags: int, gap: int = GAP) -> list[tuple[int, int]]:
    """판정이 잠깐 튄 자국을 앞 구간에 흡수한다. 다음 차의 사이렌은 절대 흡수하지 않는다.

    실제 녹음은 사이렌이 이어지는 중에도 판정이 1~2초 튄다(실측: 9~16초 siren, 17초
    normal, 18초 siren). 안 메우면 8초 사이렌이 8+1 로 쪼개지고, 번짐 보정 4초를 뺀 뒤엔
    둘 다 5초에 못 미쳐 클립이 하나도 안 나온다.

    다만 **간격으로는 못 가른다** — 번짐이 앞 run 의 끝을 4초 늘려놔서, 튄 자국이든 5초
    떨어진 다른 차든 판정상 간격이 똑같이 1 tick 이다. 대신 길이로 갈린다: 튄 자국은
    1~2 tick 이고 독립된 사이렌은 WIN 이상 이어진다. 잘못 붙이면 다음 차의 사이렌이 이
    라벨로 딸려오므로, 애매하면 안 붙이는 쪽으로 기운다.
    """
    min_own = int(WIN / TICK)
    out: list[tuple[int, int]] = []
    for a, b in runs:
        # b == n_flags 면 창 끝에서 잘린 run 이라 실제 길이를 알 수 없다 → 흡수하지 않는다.
        stray = (b - a) < min_own and b < n_flags
        if out and stray and a - out[-1][1] <= gap:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


def _span_from_run(a: int, b: int) -> tuple[float, float]:
    """사이렌 판정이 연속된 tick run [a, b) → 실제 사이렌이 울린 구간(초).

    tick i 의 판정은 오디오 [(i+1)-WIN, i+1) 초를 본 결과다. 5초 창이라 사이렌이 1초만
    걸쳐도 siren 이 뜬다(검출기는 빨리 경보하는 게 목적이라 이게 맞는 동작이다).
    그래서 판정이 켜진 시각을 그대로 쓰면 구간이 앞뒤로 최대 4초씩 번지고, 앞의 소음이나
    **다음 차의 사이렌**까지 이 라벨로 딸려온다.

      - 첫 siren tick a  → 사이렌이 창에 막 들어온 순간 → 시작 ≈ a
      - 마지막 siren tick b-1 → 사이렌이 창 앞쪽에만 남은 순간 → 끝 ≈ b - (WIN-TICK)

    확신도로는 못 가른다(실측: 사이렌 1/5초 클립 conf 0.893 > 순수 사이렌 0.850).
    """
    return a * TICK, b * TICK - (WIN - TICK)


def _windows(start_s: float, end_s: float, win: float = WIN) -> list[float]:
    """[start, end) 를 win 초씩 겹치지 않게 채운 시작시각들. 남는 꼬리는 버린다."""
    out: list[float] = []
    t = start_s
    while t + win <= end_s + 1e-9:
        out.append(t)
        t += win
    return out


def _overlaps(t0: float, taken: list[tuple[float, float]], win: float = WIN) -> bool:
    """이미 잘라간 구간과 겹치는가.

    소방서 앞에서는 구급차·소방차가 연달아 나오므로 태그 창이 서로 겹친다. 같은 오디오가
    두 라벨로 저장되면 학습이 망가지므로, 먼저 잡힌 쪽만 남긴다.
    ponytail: 선형 스캔. 세션당 클립이 수십 개라 정렬·구간트리는 과하다.
    """
    return any(t0 < b and a < t0 + win for a, b in taken)


def _read_wav(path: Path) -> np.ndarray:
    """16bit mono wav → float32 (-1, 1)."""
    with wave.open(str(path), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{path}: 16bit mono 가 아니다 (tag_siren.py 산출물이어야 함)")
        if w.getframerate() != SAMPLE_RATE:
            raise ValueError(f"{path}: {w.getframerate()}Hz — {SAMPLE_RATE}Hz 여야 한다")
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def _write_wav(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes((np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())


def _read_tags(path: Path) -> list[tuple[float, str]]:
    tags: list[tuple[float, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        t_sec, label = line.split(",")
        tags.append((float(t_sec), label.strip()))
    return tags


def _siren_spans(audio: np.ndarray) -> list[tuple[float, float]]:
    """창 오디오에서 사이렌 구간(창 로컬 초)을 찾는다.

    analyze 는 5초 롤링 버퍼를 **전역 싱글턴에 들고** 있으므로 창마다 reset() 이 필수다.
    """
    from classifier.inference import analyze, reset
    from core.types import SoundClass

    reset()
    n = int(SAMPLE_RATE * TICK)
    flags: list[bool] = []
    for start in range(0, len(audio) - n + 1, n):
        res = analyze(AudioChunk(samples=audio[start:start + n], sample_rate=SAMPLE_RATE))
        flags.append(res is not None and res["label"] is SoundClass.SIREN)
    return [_span_from_run(a, b) for a, b in _merge(_runs(flags), len(flags))]


def _cut_session(session: Path, out_root: Path, pre: float, post: float) -> tuple[list[dict], str]:
    """세션 하나 → (labels.csv 행들, 한 줄 진단).

    진단을 같이 내는 이유: 클립 0개일 때 원인이 "사이렌이 검출되지 않았다"인지 "검출은
    됐는데 5초를 못 채웠다"인지 도구가 말해주지 않으면, 매번 세션 wav 를 따로 뜯어봐야 한다.
    """
    audio = _read_wav(session / "audio.wav")
    tags = _read_tags(session / "tags.csv")
    duration = len(audio) / SAMPLE_RATE
    parts = session.name.split("_", 2)
    place = parts[2] if len(parts) > 2 else ""

    rows: list[dict] = []
    taken: list[tuple[float, float]] = []
    n_spans = n_short = 0
    for t_tag, label in tags:
        w0 = max(0.0, t_tag - pre)
        w1 = min(duration, t_tag + post)
        if w1 - w0 < WIN:
            continue

        window = audio[int(w0 * SAMPLE_RATE):int(w1 * SAMPLE_RATE)]
        for lo, hi in _siren_spans(window):
            n_spans += 1
            if hi - lo < WIN:
                n_short += 1
            for local in _windows(lo, hi):
                t0 = w0 + local
                if _overlaps(t0, taken):    # 앞 태그가 이미 가져간 구간
                    continue
                taken.append((t0, t0 + WIN))
                name = f"{session.name}_{len(rows):03d}.wav"
                _write_wav(out_root / label / name,
                           audio[int(t0 * SAMPLE_RATE):int((t0 + WIN) * SAMPLE_RATE)])
                rows.append({"clip": f"{label}/{name}", "label": label,
                             "session_id": session.name, "place": place, "t_start_sec": t0})

    if n_spans == 0:
        diag = "사이렌 구간 없음 — 검출기가 이 소리를 사이렌으로 보지 않았다"
    elif n_short == n_spans and not rows:
        diag = f"사이렌 구간 {n_spans}개가 전부 {WIN:.0f}초 미만 — 더 길게 울릴 때 태그할 것"
    else:
        diag = f"사이렌 구간 {n_spans}개" + (f" (그중 {n_short}개는 짧아 버림)" if n_short else "")
    return rows, diag


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="태깅 세션 → 차종 재학습용 5초 클립")
    p.add_argument("--sessions", default="data/siren_sessions", help="세션 폴더들의 루트")
    p.add_argument("--out", default="data/siren_clips", help="클립·labels.csv 를 쓸 위치")
    p.add_argument("--pre", type=float, default=25.0,
                   help="태그 시점에서 되감을 초 (차를 보고 키를 누르기까지의 지연)")
    p.add_argument("--post", type=float, default=10.0, help="태그 시점 이후로 볼 초")
    a = p.parse_args(argv)

    root, out_root = Path(a.sessions), Path(a.out)
    sessions = sorted(d for d in root.glob("*") if (d / "tags.csv").exists())
    if not sessions:
        print(f"[cut] 세션이 없다: {root}", file=sys.stderr)
        return 1

    rows: list[dict] = []
    for s in sessions:
        got, diag = _cut_session(s, out_root, a.pre, a.post)
        rows.extend(got)
        print(f"[cut] {s.name}: 클립 {len(got)}개 — {diag}")

    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / "labels.csv").open("w", encoding="utf-8") as f:
        f.write("clip,label,session_id,place,t_start_sec\n")
        for r in rows:
            f.write(f"{r['clip']},{r['label']},{r['session_id']},{r['place']},"
                    f"{r['t_start_sec']:.1f}\n")

    by_label: dict[str, int] = {}
    for r in rows:
        by_label[r["label"]] = by_label.get(r["label"], 0) + 1
    print(f"[cut] 총 {len(rows)}개 → {out_root}/labels.csv")
    for label, n in sorted(by_label.items()):
        print(f"        {label:<9} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
