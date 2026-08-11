"""수집 세션 → 검출기 재학습용 매니페스트 (하드 네거티브 + 사이렌 평가셋).

`main.py --collect` 로 모은 실차 오디오를 학습 리포(Airacle)로 넘기기 위한 도구.
클립을 자르지 않고 **원본 클립 경로 + 역할**만 내보낸다 — 5초 창 분할은 학습
리포의 dataset._grid(5초/1초 stride)가 이미 하고, 거기서 잘라야 학습·추론
전처리가 한 곳에서 관리된다.

## 왜 스크린이 필요한가 (비대칭 위험)

진짜 사이렌이 네거티브로 섞이면, 모델이 **사이렌을 무시하도록** 학습된다 —
오경보보다 나쁜 안전 결함이다. 반대로 네거티브를 몇 개 빠뜨리는 건 그냥 손해다.
그래서 의심스러우면 무조건 뺀다(hold). 네거티브가 되려면 두 스크린을 **동시에**
통과해야 한다:

  1. 검출 신호가 약하다 — 진짜 사이렌은 지속 발화한다(실측: 확인된 사이렌은
     연속 5틱 이상·conf 0.80 이상, 중앙값 13틱/0.98).
  2. 음향이 사이렌 프로필 밖이다 — 사이렌은 사이렌 대역에 순음 피크가 서고
     그 피크가 크게 오르내린다(wail). 확인된 사이렌 8개의 하한을 기준으로 쓴다.

사람 라벨과의 관계:
  - `not_siren` (사람이 '아님' 확인) 도 스크린을 통과해야 쓴다. 라벨이 grace 로
    엉뚱한 클립에 붙었을 수 있고, 그 한 건이 모델을 귀먹게 만들 수 있다.
  - `unlabeled` (판정 안 함) 은 스크린만 통과하면 네거티브로 쓴다. 사람이 아무
    버튼도 안 눌렀다는 건 차종을 못 봤다는 뜻이지 사이렌이 없었다는 보증은
    아니므로, 스크린이 유일한 근거다 — 그래서 기준을 사람 라벨과 똑같이 건다.
  - 차종 라벨(ambulance/police/fire/unknown)은 **평가 전용**이다. 학습에 넣지
    않는다 — 이벤트가 십수 개뿐이고 한 장소라, 넣으면 그 장소에 맞춰질 뿐이고
    '재학습이 진짜 사이렌을 안 놓치는가'를 잴 잣대가 사라진다.

사용:
  python tools/export_hardneg.py --out ../Airacle/data/realroad_manifest.json
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.types import SAMPLE_RATE
from tools.cut_clips import _read_wav

SIREN_LABELS = ("ambulance", "police", "fire", "unknown")
NOT_SIREN, UNLABELED, EXCLUDE = "not_siren", "unlabeled", "exclude"

# 스크린 임계 — 확인된 진짜 사이렌(자동 트리거)에서 뽑은 하한이라, 진짜 사이렌은
# 정의상 전부 hold 쪽으로 걸린다. 네거티브 쪽에서만 '확실히 아래'인 것을 고른다.
DET_TICKS, DET_CONF = 4, 0.90       # 이 이상이면 검출 신호가 진짜 사이렌급
F0_MIN, SWING_MIN = 515.0, 420.0    # 이 이상이면 음향이 사이렌급


def _acoustic(x: np.ndarray) -> tuple[float, float]:
    """사이렌 대역 순음 피크의 (중앙 f0, wail 스윙). 순음 프레임이 적으면 (0, 0).

    사이렌은 대역 안에 뚜렷한 순음이 서고(tonality), 그 피크 주파수가 크게
    오르내린다(wail/yelp). 노면·엔진음은 순음이 서지 않거나 서도 고정이다.
    """
    from scipy.signal import stft

    fq, _, Z = stft(x, SAMPLE_RATE, nperseg=1024, noverlap=768)
    S = np.abs(Z)
    band = (fq >= 400) & (fq <= 2000)
    Sb, fb = S[band], fq[band]
    tonality = Sb.max(axis=0) / (np.median(S, axis=0) + 1e-12)
    peak_f = fb[Sb.argmax(axis=0)]
    strong = tonality > 8
    if strong.sum() < 15:                    # 순음이 거의 없다 = 사이렌 아님
        return 0.0, 0.0
    p = peak_f[strong]
    return float(np.median(p)), float(np.percentile(p, 90) - np.percentile(p, 10))


def _max_run(flags: str) -> int:
    best = cur = 0
    for c in flags:
        cur = cur + 1 if c == "1" else 0
        best = max(best, cur)
    return best


WIN_S = 5.0     # 학습 창 길이 — 학습 리포 dataset.WIN_S 와 같아야 한다


def _detector_span(row: dict) -> list[float] | None:
    """사람이 구간을 안 적어 준 사이렌 클립의 구간을 검출 발화에서 유도한다.

    이벤트 클립은 앞에 프리롤(기본 10초)이 붙어 있다. 클립 전체를 사이렌으로 주면
    사이렌이 없는 프리롤 도로소음이 양성으로 학습돼, 줄이려던 오검출을 오히려
    키운다.

    det_flags 의 tick i 는 클립 시각 pre_roll+i 에서의 판정이고, 그 판정은 직전
    5초를 본 결과다 → 사이렌이 있을 수 있는 구간은
      [pre_roll + first + 1 - WIN,  pre_roll + last + 1].
    발화가 하나도 없으면(수동 클립) None — 유도할 근거가 없다.
    """
    flags = row["det_flags"]
    if "1" not in flags:
        return None
    first, last = flags.index("1"), len(flags) - 1 - flags[::-1].index("1")
    pre, dur = float(row["pre_roll_sec"]), float(row["duration_sec"])
    return [max(0.0, pre + first + 1 - WIN_S), min(dur, pre + last + 1)]


def _screen(row: dict, audio: np.ndarray) -> tuple[bool, str]:
    """네거티브로 써도 되나 → (통과 여부, 이유)."""
    if row["trigger"] == "manual":
        # 버튼이 먼저 눌린 클립 = 사람이 '사이렌이 들린다'고 주장한 오디오.
        # 오누름이 섞여 있고(실측 확인), 검출기는 여기서 아무 근거도 주지 못한다.
        return False, "manual(버튼 우선) — 사람 확인 필요"
    ticks, conf = _max_run(row["det_flags"]), float(row["det_conf_max"])
    if ticks >= DET_TICKS or conf >= DET_CONF:
        return False, f"검출 신호 사이렌급({ticks}틱 conf{conf:.2f})"
    f0, swing = _acoustic(audio)
    if f0 >= F0_MIN and swing >= SWING_MIN:
        return False, f"음향 사이렌급(f0 {f0:.0f}Hz swing {swing:.0f})"
    return True, f"{ticks}틱 conf{conf:.2f} f0 {f0:.0f} swing {swing:.0f}"


def _load_review(path: Path) -> dict[tuple[str, str], dict]:
    """청취 확정 라벨(collect/review.csv). 수집 중 버튼 라벨보다 **우선한다**.

    현장 라벨은 놓치거나(unlabeled) 엉뚱한 클립에 붙을 수 있다(grace 소급).
    사람이 실제로 듣고 적은 이 파일이 정본이다.
    """
    if not path.exists():
        return {}
    out: dict[tuple[str, str], dict] = {}
    lines = [l for l in path.read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.startswith("#")]
    for r in csv.DictReader(lines):
        span = None
        if r.get("t_start") and r.get("t_end"):
            span = [float(r["t_start"]), float(r["t_end"])]
        out[(r["session"], r["clip"])] = dict(label=r["label"], span=span,
                                              note=r.get("note", ""))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="수집 세션 → 재학습 매니페스트")
    p.add_argument("--sessions", default="data/collect_sessions")
    p.add_argument("--review", default="collect/review.csv",
                   help="청취 확정 라벨 (수집 중 라벨보다 우선)")
    p.add_argument("--out", required=True, help="매니페스트 JSON 경로")
    p.add_argument("--holdout", type=float, default=0.35,
                   help="네거티브 중 평가용으로 뺄 비율(클립 단위)")
    a = p.parse_args(argv)

    review = _load_review(Path(a.review))
    rows = []
    for f in sorted(glob.glob(f"{a.sessions}/*/labels.csv")):
        session = Path(f).parent
        for r in csv.DictReader(open(f, encoding="utf-8")):
            r["_session"] = session.name
            r["_wav"] = str((session / r["clip"]).resolve())
            rows.append(r)

    neg, hold, siren = [], [], []
    n_review = 0
    for r in rows:
        rec = dict(wav=r["_wav"], session=r["_session"], clip=r["clip"],
                   human_label=r["label"], trigger=r["trigger"],
                   duration_sec=float(r["duration_sec"]))
        rv = review.get((r["_session"], r["clip"]))
        if rv is not None:
            # 청취 확정본이 이긴다. 스크린은 '사람이 못 들어본 것'을 위한 대리
            # 판단일 뿐이라, 사람이 실제로 들은 순간 스크린은 할 일이 끝난다.
            n_review += 1
            rec.update(human_label=rv["label"], reviewed=True, note=rv["note"])
            if rv["label"] == EXCLUDE:
                # 사람이 듣고도 판단이 안 서는 오디오. 네거티브로 흘려보내면 안 된다
                # — '사이렌인지 모르겠다'는 '사이렌이 아니다'가 아니다.
                hold.append({**rec, "role": "hold", "why": "청취 판단 불가"})
                continue
            if rv["label"] in SIREN_LABELS:
                span = rv["span"] or _detector_span(r)
                siren.append({**rec, "role": "siren_eval", "span": span,
                              "span_source": "human" if rv["span"] else "detector"})
            else:
                neg.append({**rec, "role": "neg", "why": "청취 확정 not_siren"})
            continue
        if r["label"] in SIREN_LABELS:
            # manual = 버튼이 먼저 눌린 클립. 네거티브에 적용한 기준을 양성에도 똑같이
            # 적용한다 — 오누름이 실제로 확인됐고(실측: 3개 중 최소 1개), 이걸 평가
            # 사이렌에 넣으면 '모델이 사이렌을 놓쳤다'와 '사람이 잘못 눌렀다'가
            # 구분되지 않아 재현율 수치 자체가 거짓이 된다.
            role = "siren_suspect" if r["trigger"] == "manual" else "siren_eval"
            siren.append({**rec, "role": role, "span": _detector_span(r),
                          "span_source": "detector"})
            continue
        ok, why = _screen(r, _read_wav(r["_wav"]))
        rec["why"] = why
        (neg if ok else hold).append({**rec, "role": "neg" if ok else "hold"})

    # 클립(=이벤트) 단위로 학습/평가 분리. 같은 클립의 5초 창들은 사실상 같은
    # 소리라, 창 단위로 나누면 평가가 부풀어 개선을 실제보다 크게 본다.
    neg.sort(key=lambda d: (d["session"], d["clip"]))
    step = max(2, int(round(1.0 / max(a.holdout, 1e-6))))
    for i, d in enumerate(neg):
        d["role"] = "neg_heldout" if i % step == 0 else "neg_train"

    man = dict(
        note="실차 수집 하드 네거티브 + 사이렌 평가셋. 사이렌은 평가 전용(학습 금지).",
        screen=dict(det_ticks=DET_TICKS, det_conf=DET_CONF,
                    f0_min=F0_MIN, swing_min=SWING_MIN),
        items=neg + siren + hold,
    )
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")

    def secs(role):
        return sum(d["duration_sec"] for d in man["items"] if d["role"] == role)
    print(f"[export] → {out}  (청취 확정 반영 {n_review}클립)")
    for role in ("neg_train", "neg_heldout", "siren_eval", "siren_suspect", "hold"):
        n = sum(1 for d in man["items"] if d["role"] == role)
        print(f"  {role:12s} {n:3d}클립 {secs(role)/60:5.1f}분")
    print(f"  ※ hold 는 매니페스트에 남기되 학습·평가 어디에도 쓰지 않는다 — "
          f"사람이 들어보고 라벨을 채우면 그때 합류한다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
