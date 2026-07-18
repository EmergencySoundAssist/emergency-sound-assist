"""
approach/detector.py 단독 검증 — 합성 통과(pass-by)로 접근→멀어짐 확인.

classifier·pipeline 없이 이 모듈만 노트북에서 검증한다 (design.md TODO 충족).
실행:  cd emergency-sound-assist && python approach/test_detector.py

물리 합성(retarded-time 근사):
  차량이 속도 v로 측면거리 d 도로를 직선 통과. 관측 주파수는 시선속도로 도플러,
  진폭은 1/r 감쇠. 통과 전(approach)·후(recede)로 detector가 부호를 뒤집어야 통과.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 프로젝트 루트

from core.types import AudioChunk, Motion, SAMPLE_RATE  # noqa: E402
from approach.detector import ApproachDetector            # noqa: E402

C = 343.0  # 음속 m/s


def synth_passby(sr=SAMPLE_RATE, f0=700.0, v_kmh=60.0, d=8.0, dur=10.0, snr_db=20.0):
    """정지 톤 f0를 속도 v·측면거리 d의 통과 신호로 합성. (n,) float32."""
    v = v_kmh / 3.6
    t = np.arange(0.0, dur, 1.0 / sr)
    t0 = dur / 2.0
    x = v * (t - t0)                       # 도로상 위치 (통과 시 x=0)
    r = np.sqrt(d ** 2 + x ** 2)
    drdt = v * x / r                       # dr/dt
    v_closing = -drdt                      # >0: 접근 중
    f_obs = f0 * C / (C - v_closing)       # 도플러
    amp = d / r                            # 1/r 감쇠 (최근접 시 ~1)
    sig = amp * np.sin(np.cumsum(2 * np.pi * f_obs / sr))
    # 잡음 주입
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(sig.size)
    sp = np.mean(sig ** 2) + 1e-12
    noise *= np.sqrt(sp / (10 ** (snr_db / 10)) / (np.mean(noise ** 2) + 1e-12))
    return (sig + noise).astype(np.float32), t0


def run(label, **kw):
    sig, t0 = synth_passby(**kw)
    det = ApproachDetector()
    chunk_n = SAMPLE_RATE                  # 1초 청크
    print(f"\n=== {label} (통과 t0={t0:.1f}s) ===")
    timeline = []
    for i, start in enumerate(range(0, sig.size - chunk_n + 1, chunk_n)):
        chunk = AudioChunk(samples=sig[start:start + chunk_n])
        m = det.update(chunk).motion
        t_end = (start + chunk_n) / SAMPLE_RATE
        phase = "접근구간" if t_end <= t0 else "이탈구간"
        timeline.append((t_end, phase, m))
        print(f"  t={t_end:4.1f}s [{phase}]  ->  {m.value}")

    # 판정: 통과 후(이탈구간)는 RECEDING이 우세해야, 통과 전 후반부는 APPROACHING이 나와야.
    approaching = [m for t, p, m in timeline if p == "접근구간" and m == Motion.APPROACHING]
    receding = [m for t, p, m in timeline if p == "이탈구간" and m == Motion.RECEDING]
    ok = len(approaching) >= 1 and len(receding) >= 2
    print(f"  결과: 접근 검출 {len(approaching)}회 · 이탈 검출 {len(receding)}회  ->  "
          f"{'[PASS]' if ok else '[CHECK]'}")
    return ok


def test_speed_level_edges():
    """음량 기울기 크기 → 접근 빠르기 1~5 (경계 포함)."""
    from approach.detector import _speed_level
    assert _speed_level(0.0) == 1
    assert _speed_level(0.30) == 2
    assert _speed_level(0.60) == 3
    assert _speed_level(0.90) == 4
    assert _speed_level(1.30) == 5
    assert _speed_level(9.9) == 5           # 클램프


def test_update_carries_speed_level_field():
    """update() 가 ApproachResult.speed_level 을 채운다(무음이면 None)."""
    from core.types import ApproachResult
    det = ApproachDetector()
    r = det.update(AudioChunk(samples=np.zeros(SAMPLE_RATE, dtype=np.float32)))
    assert isinstance(r, ApproachResult)
    assert r.speed_level is None           # 무음 → 접근 아님 → None


if __name__ == "__main__":
    results = [
        run("기본 60km/h·8m·clean", snr_db=40.0),
        run("60km/h·8m·20dB 잡음", snr_db=20.0),
        run("40km/h·12m·20dB 잡음", v_kmh=40.0, d=12.0, snr_db=20.0),
    ]
    print(f"\n총 {sum(results)}/{len(results)} 시나리오 PASS")
