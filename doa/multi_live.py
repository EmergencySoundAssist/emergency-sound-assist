"""다중 음원 방향 추정 — 실시간 루프 테스트.

짧은 창을 반복 캡처해 방향을 계속 출력한다. 소리를 옮겨가며 각도가 따라오는지
눈으로 확인하는 용도. 소리가 작으면(조용하면) 엉뚱한 각도 대신 "조용함"을 띄운다.

반사·잡음으로 방향이 ±180° 튀는 것을 막기 위해 **시간 다수결+신뢰도 게이팅**을
기본 적용한다(doa/tracking.py). 끄고 옛 즉시반응 동작과 비교하려면 `--no-smooth`.

스무딩 범위/지연 주의:
  · **검출**(소리 왔다 = rms 임계)은 즉시 그대로 — 지연 0.
  · **화살표(방향)** 는 안정화되며, 새 음원의 첫 화살표는 신뢰 프레임이 모일 때까지
    최대 (SMOOTH_MIN_FRAMES-1) hop(기본 ~0.1초) 늦게 뜬다. 사이렌(수 초)엔 무시 가능.
  · 안정화는 **대표(primary) 방향만** 적용. 동시 음원의 부방향은 이번 프레임값(best-effort)
    이라 여전히 튈 수 있다.

실행 (airacle 환경):
    python -m doa.multi_live
    python -m doa.multi_live --led
    python -m doa.multi_live --no-smooth          # 스무딩 끄고 비교
    python -m doa.multi_live --conf-min 8 --num 2

Ctrl+C 로 종료.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from doa import config, led_ring
from doa.estimator import angle_to_direction
from doa.multi_source import (
    FS_DEFAULT,
    estimate_multiple_directions,
    find_respeaker_device,
)
from doa.tracking import DirectionTracker


def _compass_bar(angles_deg, width: int = 36) -> str:
    """0~360°를 width 칸 막대에 표시 (소리 방향에 '●')."""
    cells = ["·"] * width
    for a in angles_deg:
        cells[int(a % 360 / 360 * width) % width] = "●"
    return "[" + "".join(cells) + "]"


def main() -> None:
    ap = argparse.ArgumentParser()
    # 기본값은 doa/config.py 에서 가져오고, 플래그를 주면 이번 실행만 덮어쓴다.
    ap.add_argument("--threshold", type=float, default=config.THRESHOLD,
                    help=f"이 RMS 미만이면 '조용함' (config={config.THRESHOLD})")
    ap.add_argument("--window", type=float, default=config.WINDOW,
                    help=f"분석 창 길이(초) — 길수록 방향 해상도↑ (config={config.WINDOW})")
    ap.add_argument("--hop", type=float, default=config.HOP,
                    help=f"갱신 간격(초) — 짧을수록 빠름 (config={config.HOP})")
    ap.add_argument("--num", type=int, default=config.NUM_SRC,
                    help=f"최대 동시 음원 수 (config={config.NUM_SRC})")
    ap.add_argument("--height", type=float, default=config.HEIGHT_RATIO,
                    help=f"2번째 peak 임계(최대 대비). 낮출수록 약한 음원도 (config={config.HEIGHT_RATIO})")
    ap.add_argument("--min-sep", type=float, default=config.MIN_SEP_DEG,
                    help=f"두 음원 최소 각도 간격(도) (config={config.MIN_SEP_DEG})")
    ap.add_argument("--algo", default=config.ALGO,
                    help=f"DoA 알고리즘 SRP/MUSIC (config={config.ALGO})")
    ap.add_argument("--led", action=argparse.BooleanOptionalAction, default=config.LED,
                    help=f"감지 방향을 LED 링에 점등 (--led/--no-led, config={config.LED})")
    ap.add_argument("--hold", type=float, default=config.HOLD,
                    help=f"감지 후 LED 유지 시간(초) (config={config.HOLD})")
    ap.add_argument("--smooth", action=argparse.BooleanOptionalAction, default=config.SMOOTH,
                    help=f"시간 다수결+신뢰도 게이팅으로 방향 튐 억제 "
                         f"(--smooth/--no-smooth, config={config.SMOOTH})")
    ap.add_argument("--conf-min", type=float, default=config.CONF_MIN,
                    help=f"이 신뢰도(주엽 우월도) 미만이면 '방향 불확실' (config={config.CONF_MIN})")
    args = ap.parse_args()

    import sounddevice as sd

    dev = find_respeaker_device(sd)
    if dev is None:
        print("ReSpeaker(6채널)를 못 찾음. sd.query_devices() 확인.")
        return

    ring = None
    if args.led:
        ring = led_ring.find()
        if ring is None:
            print("LED 링을 못 찾음(pyusb 확인) — LED 없이 진행.")
        else:
            ring.set_brightness(config.LED_BRIGHTNESS)
    fs = FS_DEFAULT
    win_n = int(args.window * fs)
    hop_n = max(1, int(args.hop * fs))
    rolling = np.zeros((win_n, 4), dtype="float32")  # 최근 window 만큼의 ch1~4

    # 시간 견고화: 최근 SMOOTH_WIN 초의 신뢰 프레임을 원형 다수결 (대표 방향만 안정화)
    tracker = None
    if args.smooth:
        smooth_frames = max(1, round(config.SMOOTH_WIN / args.hop))
        tracker = DirectionTracker(
            maxlen=smooth_frames, conf_min=args.conf_min,
            min_frames=config.SMOOTH_MIN_FRAMES,
        )
    print(f"ReSpeaker device={dev} | window={args.window}s hop={args.hop}s "
          f"(~{1/args.hop:.0f}Hz) | LED={'on' if ring else 'off'} | "
          f"smooth={'on' if tracker else 'off'} | Ctrl+C 종료\n")

    multi_count = 0
    lit = False           # 지금 LED 가 켜져 있는지
    last_seen = 0.0       # 마지막 감지 시각(monotonic)
    try:
        # 연속 스트림: 끊김 없이 hop 단위로 읽고, 최근 window 를 분석 (슬라이딩 윈도우)
        with sd.InputStream(samplerate=fs, channels=6, device=dev,
                            dtype="float32", blocksize=hop_n) as stream:
            while True:
                block, overflowed = stream.read(hop_n)        # ~hop 초 만큼 블록
                rolling = np.roll(rolling, -hop_n, axis=0)
                rolling[-hop_n:] = np.asarray(block)[:, 1:5]   # ch1~4 만
                now = time.monotonic()

                rms = float(np.sqrt(np.mean(rolling ** 2)))
                results = []
                conf = 0.0
                if rms >= args.threshold:
                    if tracker is not None:
                        results, conf = estimate_multiple_directions(
                            rolling, fs=fs, num_src=args.num, algo=args.algo,
                            height_ratio=args.height, min_sep_deg=args.min_sep,
                            with_confidence=True,
                        )
                    else:
                        results = estimate_multiple_directions(
                            rolling, fs=fs, num_src=args.num, algo=args.algo,
                            height_ratio=args.height, min_sep_deg=args.min_sep,
                        )

                # 표시할 방향 결정: 스무딩 ON 이면 대표(primary)만 다수결로 안정화하고
                # 부방향은 이번 프레임값(best-effort). 신뢰 프레임이 모자라면 '방향 불확실'.
                uncertain = False
                if tracker is not None:
                    primary = results[0][0] if results else None
                    tr = tracker.update(primary, conf)
                    if tr.angle is not None:
                        angles = [tr.angle] + [a for a, _ in results[1:]]
                    else:
                        angles = []
                        uncertain = bool(results)  # peak 는 있는데 방향 확정 실패
                else:
                    angles = [a for a, _ in results]

                # '실감지'는 이번 프레임에 실제 peak 가 있을 때만(live). 스무딩으로 tr.angle 이
                # 무음 중에도 잠깐 남을 수 있는데, 그땐 last_seen 을 갱신하지 않아야 hold 가
                # '마지막 실감지' 기준으로 동작한다 (LED-on 이 hold 를 초과하지 않게).
                live = bool(results)

                if angles and live:
                    # 방향 확정 → LED 점등 + 유지 타이머 갱신
                    if ring:
                        ring.show_directions(angles)
                        lit = True
                    last_seen = now
                    label = "  ".join(
                        f"{a:3.0f}°({angle_to_direction(a).value})" for a in angles
                    )
                    conf_s = f" conf={conf:.1f}" if tracker is not None else ""
                    lag = " ⚠느림" if overflowed else ""
                    line = f"{_compass_bar(angles)}  {label}  rms={rms:.3f}{conf_s}{lag}"
                    if len(angles) >= 2:
                        multi_count += 1
                        print(f"\r#{multi_count:<3} ★다중({len(angles)})  {line}        ")
                    else:
                        print(f"\r{line}        ", end="", flush=True)
                else:
                    # 방향 없음 → hold 동안은 직전 불빛 유지, 지나면 끔
                    held = lit and (now - last_seen) <= args.hold
                    if ring and lit and not held:
                        ring.off()
                        lit = False
                    if uncertain:
                        state = f"방향 불확실 (감지됨 conf={conf:.1f})"
                    else:
                        state = "유지중" if held else "조용함"
                    print(f"\r{_compass_bar([])}  {state} (rms={rms:.4f})        ", end="", flush=True)
    except KeyboardInterrupt:
        if ring:
            ring.off()
            ring.close()
        print("\n종료")


if __name__ == "__main__":
    main()
