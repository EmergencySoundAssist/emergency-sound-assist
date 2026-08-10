"""수집 세션(collect/) → 차종 재학습용 5초 클립.

main.py --collect 가 남긴 이벤트 클립(clips/*.wav + labels.csv)을 읽어,
클립 안에서 검출기로 실제 사이렌 구간을 확정하고 5초씩 자른다. 구간 검출·창
분할·번짐 보정 로직은 tools/cut_clips.py 의 것을 그대로 쓴다(계약 공유).

tag_siren 세션과 다른 점 세 가지:
  - 이벤트 클립은 이미 사이렌 하나를 중심으로 잘려 있다 → 태그 시점 되감기가 없다.
  - **수동(manual) 클립은 검출기가 놓친 사이렌**이다. 검출기 구간으로 5초 창이
    하나도 안 나오면(구간 없음뿐 아니라 1~8틱 부분 검출로 짧은 경우 포함) 버튼
    시점(pre_roll) 언저리부터 **고정 상한(FIXED_CAP)까지만** 잘라 refined=fixed 로
    표시한다 — 상한이 없으면 오누름·사이렌 종료 후의 순수 도로소음까지 라벨을
    달고 학습셋에 들어간다.
  - 연속 차량이 auto_tail 안에 이어져 **한 클립으로 병합**될 수 있다. 그때는
    labels.csv 의 presses(녹음 중 누른 시각:라벨 목록)로 구간마다 라벨을 나눠
    붙인다. 누름이 서로 다른 라벨 2개 이상일 때만 발동하고, 매칭 안 되는 구간은
    버린다(잘못 붙이느니 버린다 — cut_clips 의 원칙과 동일).

사용:
  python tools/cut_collect.py                    # data/collect_sessions/* → data/collect_clips/
  python tools/cut_collect.py --lead 2.5         # fixed 폴백 창의 버튼 앞 여유 조정

산출:
  data/collect_clips/{ambulance,police,fire,unknown}/<세션>_<클립>_<번호>.wav
  data/collect_clips/labels.csv  (clip,label,trigger,refined,session_id,place,src,t_start_sec)

labels.csv 는 매 실행마다 새로 쓴다. train/test 는 session_id 또는 place 단위로
나눌 것 — 같은 출동에서 잘린 클립들은 사실상 같은 소리다(cut_clips.py 와 동일 규칙).
unlabeled 클립은 자르지 않는다(사람 라벨이 없는 오디오는 학습에 못 쓴다) — 개수만
알려주니, 세션 폴더에서 다시 듣고 labels.csv 의 label 을 채운 뒤 재실행하면 된다.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.types import SAMPLE_RATE
from tools.cut_clips import WIN, _read_wav, _siren_spans, _windows, _write_wav

UNLABELED = "unlabeled"
NOT_SIREN = "not_siren"     # 오검출 확인 클립 — 검출기 hard-negative 로만 자른다
FIXED_CAP = 15.0        # fixed 폴백이 버튼 언저리에서 가져갈 최대 초


def _fixed_span(duration: float, pre_roll: float, lead: float,
                cap: float = FIXED_CAP) -> tuple[float, float]:
    """검출기 창이 안 나온 수동 클립의 폴백 창.

    버튼을 누른 시점(pre_roll)엔 사이렌이 이미 들리고 있었다 — lead 초만 앞으로
    물러나 거기서부터 최대 cap 초. 클립 끝까지 쓰면 사이렌이 먼저 끝났을 때
    나머지 주행소음이 같은 라벨로 딸려온다.
    """
    lo = max(0.0, pre_roll - lead)
    return lo, min(duration, lo + cap)


def _parse_presses(s: str) -> list[tuple[float, str]]:
    """'12.0:ambulance|30.5:fire' → [(12.0, 'ambulance'), (30.5, 'fire')]."""
    out: list[tuple[float, str]] = []
    for part in (s or "").split("|"):
        part = part.strip()
        if not part:
            continue
        t, lab = part.split(":", 1)
        out.append((float(t), lab))
    return out


def _assign_spans(spans, presses, default_label):
    """구간별 라벨 배정 → [(lo, hi, label|None)].

    서로 다른 라벨이 2개 이상 눌렸다는 건 연속 차량이 한 클립으로 병합됐다는
    뜻이다. 누름은 항상 그 차의 사이렌 시작 **이후**(차를 보고 나서) 오므로,
    구간 순서대로 '시작+1초 이후의 가장 이른 미사용 누름'을 짝짓는다.
    매칭 실패 구간은 None — 어느 차인지 모르는 오디오는 버린다.
    """
    labels = {lab for _, lab in presses}
    if len(labels) < 2:
        return [(lo, hi, default_label) for lo, hi in spans]
    presses = sorted(presses)
    out, used = [], set()
    for lo, hi in sorted(spans):
        pick = None
        for i, (t, lab) in enumerate(presses):
            if i not in used and t >= lo + 1.0:
                pick = (i, lab)
                break
        if pick is None:
            out.append((lo, hi, None))
        else:
            used.add(pick[0])
            out.append((lo, hi, pick[1]))
    return out


def _cut_session(session: Path, out_root: Path, lead: float) -> tuple[list[dict], str]:
    """세션 하나 → (labels.csv 행들, 한 줄 진단). cut_clips 와 같은 이유로 진단을 남긴다."""
    parts = session.name.split("_", 2)
    place = parts[2] if len(parts) > 2 else ""
    rows: list[dict] = []
    n_unlabeled = n_zero = n_dropped = 0

    with (session / "labels.csv").open(encoding="utf-8") as f:
        events = list(csv.DictReader(f))

    for ev in events:
        if ev["label"] == UNLABELED:
            n_unlabeled += 1
            continue
        audio = _read_wav(session / ev["clip"])
        duration = len(audio) / SAMPLE_RATE
        tagged = _assign_spans(_siren_spans(audio),
                               _parse_presses(ev.get("presses", "")), ev["label"])
        n_dropped += sum(1 for _, _, lab in tagged if lab is None)
        wins = [(t0, lab, "detector")
                for lo, hi, lab in tagged if lab is not None
                for t0 in _windows(lo, hi)]
        # 수동 = 검출기가 놓친 사이렌. 검출기 경로로 창이 0개면(구간 없음·부분
        # 검출로 5초 미만 모두 포함) 버튼 시점 기준 고정 창으로 폴백한다.
        # not_siren 은 예외 — 검출기가 실제로 속은 구간만 negative 로 가치가 있고,
        # 폴백까지 하면 그냥 도로소음이 not_siren/ 을 채운다.
        if not wins and ev["trigger"] == "manual" and ev["label"] != NOT_SIREN:
            lo, hi = _fixed_span(duration, float(ev["pre_roll_sec"]), lead)
            wins = [(t0, ev["label"], "fixed") for t0 in _windows(lo, hi)]
        if not wins:
            n_zero += 1
            continue

        stem = Path(ev["clip"]).stem
        for t0, label, refined in wins:
            name = f"{session.name}_{stem}_{len(rows):03d}.wav"
            _write_wav(out_root / label / name,
                       audio[int(t0 * SAMPLE_RATE):int((t0 + WIN) * SAMPLE_RATE)])
            rows.append({
                "clip": f"{label}/{name}", "label": label,
                "trigger": ev["trigger"], "refined": refined,
                "session_id": session.name, "place": place,
                "src": ev["clip"], "t_start_sec": round(t0, 1),
            })

    diag = f"이벤트 {len(events)}개"
    if n_unlabeled:
        diag += f" (라벨 없음 {n_unlabeled} — 자르지 않음)"
    if n_zero:
        diag += f" (창 0개 {n_zero} — 구간이 없거나 {WIN:.0f}초 미만)"
    if n_dropped:
        diag += f" (누름 매칭 실패로 버린 구간 {n_dropped})"
    return rows, diag


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="수집 세션 → 차종 재학습용 5초 클립")
    p.add_argument("--sessions", default="data/collect_sessions", help="수집 세션 루트")
    p.add_argument("--out", default="data/collect_clips", help="클립·labels.csv 를 쓸 위치")
    p.add_argument("--lead", type=float, default=2.5,
                   help="fixed 폴백에서 버튼 시점 앞으로 물러날 초")
    a = p.parse_args(argv)

    root, out_root = Path(a.sessions), Path(a.out)
    # clips/ 가 있어야 수집 세션이다 — labels.csv 만 보면 이 도구의 산출 폴더까지
    # 세션으로 착각해 다시 읽는다(둘 다 labels.csv 를 갖는다).
    sessions = sorted(d for d in root.glob("*")
                      if (d / "labels.csv").exists() and (d / "clips").is_dir())
    if not sessions:
        print(f"[cut] 세션이 없다: {root}", file=sys.stderr)
        return 1

    rows: list[dict] = []
    for s in sessions:
        got, diag = _cut_session(s, out_root, a.lead)
        rows.extend(got)
        print(f"[cut] {s.name}: 클립 {len(got)}개 — {diag}")

    out_root.mkdir(parents=True, exist_ok=True)
    cols = ["clip", "label", "trigger", "refined", "session_id", "place", "src", "t_start_sec"]
    with (out_root / "labels.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    by_label: dict[str, int] = {}
    for r in rows:
        by_label[r["label"]] = by_label.get(r["label"], 0) + 1
    print(f"[cut] 총 {len(rows)}개 → {out_root}/labels.csv")
    for label, n in sorted(by_label.items()):
        print(f"        {label:<9} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
