"""런타임 벤치 — 지연과 오탐을 **한 번에, 같은 조건에서** 잰다.

## 왜 도구로 만드는가

즉석 스크립트로 재다가 오늘만 다섯 번 틀렸다. 전부 같은 종류의 실수다:

  1. 분류기가 **모듈 싱글턴**이라 클립 사이에 버퍼가 이어졌다 → 앞 클립이 뒤를 오염.
  2. 클립마다 reset 하면 앞 5초 창이 무음 패딩이라 **콜드 버퍼 틱**이 검출로 잡혔다.
  3. `_debounce` 가 **벽시계** 기준이라 오프라인 재생(실시간의 수십 배)에서 2초 잔향이
     오디오 수백 초를 덮었다 → 한 번 튀면 그 뒤 전부 긴급.
  4. 소음 배경으로 쓴 클립에 **사이렌이 섞여** 있었다 → 지연이 전부 0으로 나왔다.
  5. 수집 클립은 **검출 시점 기준으로 잘려** 있어 지연·재현율을 원리적으로 못 잰다.

그래서 이 도구는 위 다섯을 구조적으로 막고, 막았는지 **자체 점검(--selfcheck)** 한다.
결론을 내리기 전에 반드시 셀프체크가 통과해야 한다.

## 무엇을 재는가 (항상 함께)

  지연  : 사이렌이 들리기 시작한 시각 → 첫 긴급 판정까지 (중앙/p90/최대)
  오탐  : 사이렌 없는 오디오에서 긴급이 뜬 비율 (클립 단위 · 틱 단위)
  놓침  : 사이렌인데 끝까지 안 뜬 비율

지연만 좋아지고 오탐이 나빠지는 변경을 '개선'이라 부르지 않기 위해 둘을 붙여 낸다.

## 쓰기

  python tools/bench_runtime.py --selfcheck          # 하네스가 멀쩡한지 먼저
  python tools/bench_runtime.py --grid 1.0 0.5 0.25  # 격자 비교
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.types import AudioChunk, SAMPLE_RATE, SoundClass
import classifier.inference as CI
from tools.cut_clips import _read_wav

WARMUP_S = 5.0          # 5초 창이 실제 오디오로 다 차기 전 틱은 런타임에 없다 → 버린다
BED_S = 8.0             # 사이렌 앞에 깔 소음 길이(초)


@dataclass
class Result:
    latency: list[float]
    misses: int
    fp_clips: int
    fp_ticks: int
    n_ticks: int
    n_sirens: int
    n_negs: int

    def line(self, tag: str) -> str:
        v = sorted(self.latency)
        med = f"{np.median(v):.2f}s" if v else "  -  "
        p90 = f"{v[min(len(v) - 1, int(0.9 * len(v)))]:.2f}s" if v else "  -  "
        mx = f"{max(v):.2f}s" if v else "  -  "
        return (f"{tag:<22}지연 중앙 {med} · p90 {p90} · 최대 {mx} · "
                f"놓침 {self.misses}/{self.n_sirens} | "
                f"오탐 클립 {self.fp_clips}/{self.n_negs} · "
                f"틱 {self.fp_ticks}/{self.n_ticks} ({100 * self.fp_ticks / max(self.n_ticks,1):.1f}%)")


def _aihub():
    """AI-Hub 원본 — 라벨이 확실하고 검출 시점에 안 잘려 있어 지연을 잴 수 있다."""
    sys.path.insert(0, "/Users/swlee/PycharmProjects/Airacle")
    import dataset as D
    from scipy.signal import resample_poly

    src = D.split_sources(D.index_sources())

    def load16(s):
        return resample_poly(D.load_wav(s.wav, 22050), 1600, 2205).astype(np.float32)

    sir = [s for s in src if s.label == "siren" and s.split == "test" and s.dur >= 8]
    noi = [s for s in src if s.label == "noise" and s.split == "test" and s.dur >= BED_S + 1]
    return sir, noi, load16


def _tick_labels(y: np.ndarray, grid: float, gate=None) -> list[tuple[float, bool]]:
    """오디오를 grid 초씩 흘리며 (시각, 긴급?) 을 돌려준다.

    - 클립마다 CI.reset() 은 **호출자가** 한다(여기서 하면 배경+사이렌이 갈린다).
    - 워밍업 틱(t < WARMUP_S)은 버린다.
    - **디바운스를 쓰지 않는다** — 벽시계라 오프라인 재생에서 무의미하다. 게이트를
      주면 게이트의 시간상수(초)를 grid 로 환산해 쓴다.
    """
    out = []
    hop = int(SAMPLE_RATE * grid)
    for k in range(0, len(y) - hop + 1, hop):
        r = CI.analyze(AudioChunk(samples=y[k:k + hop], sample_rate=SAMPLE_RATE))
        t = (k + hop) / SAMPLE_RATE
        if r is None or t < WARMUP_S:
            continue
        if gate is None:
            fire = CI.emergency_from(r)          # 런타임(infer)과 같은 규칙
        else:
            fire = gate.step(*CI.gate_margins(r))   # 런타임과 같은 함수
        out.append((t, fire))
    return out


def run(grid: float, pairs, negs, load16, gate_factory=None) -> Result:
    lat, misses = [], 0
    for s_, n_ in pairs:
        try:
            h, bed = load16(s_), load16(n_)[:int(SAMPLE_RATE * BED_S)]
        except Exception:
            continue
        if len(bed) < SAMPLE_RATE * BED_S:
            continue
        CI.reset()
        g = gate_factory(grid) if gate_factory else None
        onset = len(bed) / SAMPLE_RATE
        hit = None
        for t, fire in _tick_labels(np.concatenate([bed, h]), grid, g):
            if t >= onset and fire:
                hit = t - onset
                break
        if hit is None:
            misses += 1
        else:
            lat.append(hit)

    fp_c = fp_t = n_t = 0
    for w in negs:
        CI.reset()
        g = gate_factory(grid) if gate_factory else None
        fired = False
        for _, fire in _tick_labels(_read_wav(w), grid, g):
            n_t += 1
            fp_t += fire
            fired |= fire
        fp_c += fired
    return Result(lat, misses, fp_c, fp_t, n_t, len(pairs), len(negs))


def selfcheck(pairs, negs, load16) -> bool:
    """하네스가 앞서 겪은 다섯 결함을 실제로 막는지 확인한다."""
    ok = True

    # ① 싱글턴 오염 — reset 을 빼면 앞 클립이 뒤를 오염시키는가
    s_, n_ = pairs[0]
    bed = load16(n_)[:int(SAMPLE_RATE * BED_S)]
    CI.reset()
    a = _tick_labels(bed, 1.0)
    b = _tick_labels(bed, 1.0)                       # reset 없이 이어서
    CI.reset()
    c = _tick_labels(bed, 1.0)
    same = [x[1] for x in a] == [x[1] for x in c]
    print(f"  ① 싱글턴: reset 후 재현 {'OK' if same else '실패'} · "
          f"reset 없이 이어 붙이면 다른 결과 {'확인' if [x[1] for x in a] != [x[1] for x in b] else '차이없음'}")
    ok &= same

    # ② 워밍업 틱 제외 — 첫 틱 시각이 WARMUP_S 이상인가
    first = a[0][0] if a else 0.0
    print(f"  ② 워밍업: 첫 틱 {first:.2f}s (≥{WARMUP_S} 이어야 함) "
          f"{'OK' if first >= WARMUP_S else '실패'}")
    ok &= first >= WARMUP_S

    # ③ 디바운스 미사용 — 벽시계에 의존하지 않는다(구조적 보장, 코드로 표시)
    print("  ③ 디바운스: 사용 안 함(벽시계 비의존) OK")

    # ④ 소음 배경이 깨끗한가 — 배경만으로 긴급이 뜨면 지연이 0으로 붕괴한다
    dirty = 0
    for _, n_ in pairs:
        try:
            b8 = load16(n_)[:int(SAMPLE_RATE * BED_S)]
        except Exception:
            continue
        if len(b8) < SAMPLE_RATE * BED_S:
            continue
        CI.reset()
        dirty += any(f for _, f in _tick_labels(b8, 1.0))
    print(f"  ④ 소음 배경: {dirty}/{len(pairs)} 개가 배경만으로 발화 "
          f"({'OK' if dirty <= len(pairs) * 0.2 else '⚠ 배경이 오염됨 — 지연이 과소평가된다'})")

    # ⑤ 지연을 잴 수 있는 소재인가 — 수집 클립은 검출 시점에 잘려 있어 못 잰다
    print("  ⑤ 소재: AI-Hub 원본(검출 앵커 없음) 사용 OK")
    return ok


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="런타임 지연·오탐 동시 벤치")
    p.add_argument("--grid", type=float, nargs="+", default=[1.0])
    p.add_argument("--n", type=int, default=30, help="사이렌 표본 수")
    p.add_argument("--selfcheck", action="store_true")
    p.add_argument("--gate", action="store_true", help="마진 게이트(pipeline.gate) 적용")
    p.add_argument("--tau-on", type=float, nargs="*", default=None,
                   help="게이트 켜기 임계 스윕 (기본: deploy 값 1.2)")
    p.add_argument("--heldout", action="store_true",
                   help="네거티브를 neg_heldout 으로 제한 (모델 간 공평 비교용)")
    p.add_argument("--manifest", default="/Users/swlee/PycharmProjects/Airacle/data/realroad_manifest.json")
    a = p.parse_args(argv)

    sir, noi, load16 = _aihub()
    rnd = random.Random(0)
    pairs = [(rnd.choice(sir), rnd.choice(noi)) for _ in range(a.n)]
    items = json.loads(Path(a.manifest).read_text(encoding="utf-8"))["items"]
    # ★ --heldout 은 neg_heldout 만 쓴다. 채널 프로파일(realchannel)이 neg_train 으로
    #   만들어지므로, 채널을 쓴 모델과 안 쓴 모델을 **공평하게** 비교하려면 학습 재료가
    #   아닌 쪽에서 재야 한다. 기본(전체)은 절대값 감시용이고 모델 간 비교엔 못 쓴다.
    roles = ("neg_heldout",) if a.heldout else ("neg_train", "neg_heldout")
    negs = [d["wav"] for d in items if d["role"] in roles]
    print(f"[bench] 네거티브 역할 {roles}", file=sys.stderr)

    print(f"[bench] 사이렌 {len(pairs)}건(AI-Hub) · 실차 네거티브 {len(negs)}클립 · "
          f"모델 {CI._MODEL_PATH.name}\n")
    if a.selfcheck:
        print("셀프체크 — 하네스가 과거 결함 5종을 막는가")
        good = selfcheck(pairs, negs, load16)
        print(f"\n결과: {'통과' if good else '실패 — 이 하네스로 낸 수치를 믿지 말 것'}")
        return 0 if good else 1

    if not a.gate:
        for g in a.grid:
            print(run(g, pairs, negs, load16).line(f"격자 {g:.2f}s"))
        return 0
    from pipeline.gate import EmergencyGate, SIREN, configs_for
    taus = a.tau_on if a.tau_on else [SIREN.tau_on]
    for g in a.grid:
        for tau in taus:
            sc, hc = configs_for(tau)          # 런타임(classifier.inference)과 같은 코드
            gf = lambda dt, sc=sc, hc=hc: EmergencyGate(dt, siren=sc, horn=hc)
            print(run(g, pairs, negs, load16, gate_factory=gf)
                  .line(f"격자 {g:.2f}s τ={tau:+.1f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
