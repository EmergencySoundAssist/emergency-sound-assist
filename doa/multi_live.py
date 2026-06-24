"""다중 음원 방향 추정 — 실시간 루프 테스트.

짧은 창을 반복 캡처해 방향을 계속 출력한다. 소리를 옮겨가며 각도가 따라오는지
눈으로 확인하는 용도. 소리가 작으면(조용하면) 엉뚱한 각도 대신 "조용함"을 띄운다.

실행 (airacle 환경):
    python -m doa.multi_live
    python -m doa.multi_live --threshold 0.02 --window 0.4 --num 2

Ctrl+C 로 종료.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from doa import config, led_ring
from doa.multi_source import (
    FS_DEFAULT,
    estimate_multiple_directions,
    find_respeaker_device,
)


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
    print(f"ReSpeaker device={dev} | window={args.window}s hop={args.hop}s "
          f"(~{1/args.hop:.0f}Hz) | LED={'on' if ring else 'off'} | Ctrl+C 종료\n")

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
                if rms >= args.threshold:
                    results = estimate_multiple_directions(
                        rolling, fs=fs, num_src=args.num, algo=args.algo,
                        height_ratio=args.height, min_sep_deg=args.min_sep,
                    )
                angles = [a for a, _ in results]

                if angles:
                    # 새 감지 → LED 점등 + 유지 타이머 갱신
                    if ring:
                        ring.show_directions(angles)
                        lit = True
                    last_seen = now
                    label = "  ".join(f"{a:3.0f}°({d.value})" for a, d in results)
                    lag = " ⚠느림" if overflowed else ""
                    line = f"{_compass_bar(angles)}  {label}  rms={rms:.3f}{lag}"
                    if len(angles) >= 2:
                        multi_count += 1
                        print(f"\r#{multi_count:<3} ★다중({len(angles)})  {line}        ")
                    else:
                        print(f"\r{line}        ", end="", flush=True)
                else:
                    # 감지 없음 → hold 동안은 직전 불빛 유지, 지나면 끔
                    held = lit and (now - last_seen) <= args.hold
                    if ring and lit and not held:
                        ring.off()
                        lit = False
                    state = "유지중" if held else "조용함"
                    print(f"\r{_compass_bar([])}  {state} (rms={rms:.4f})        ", end="", flush=True)
    except KeyboardInterrupt:
        if ring:
            ring.off()
            ring.close()
        print("\n종료")


if __name__ == "__main__":
    main()
