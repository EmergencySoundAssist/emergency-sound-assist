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

import numpy as np

from doa import led_ring
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
    ap.add_argument("--threshold", type=float, default=0.01,
                    help="이 RMS 미만이면 '조용함' 처리 (기본 0.01)")
    ap.add_argument("--window", type=float, default=0.4,
                    help="분석 창 길이(초) — 길수록 방향 해상도↑ (기본 0.4)")
    ap.add_argument("--hop", type=float, default=0.1,
                    help="갱신 간격(초) — 짧을수록 자주 갱신(빠름). 기본 0.1 = 약 10Hz")
    ap.add_argument("--num", type=int, default=2, help="최대 동시 음원 수")
    ap.add_argument("--height", type=float, default=0.5,
                    help="2번째 peak 인정 임계(최대치 대비). 낮출수록 약한 음원도 잡음 (기본 0.5)")
    ap.add_argument("--min-sep", type=float, default=30.0,
                    help="두 음원 최소 각도 간격(도). 낮출수록 가까운 음원 분리 (기본 30)")
    ap.add_argument("--algo", default="SRP",
                    help="DoA 알고리즘: SRP(기본) 또는 MUSIC (다중 음원엔 MUSIC 유리)")
    ap.add_argument("--led", action="store_true",
                    help="감지된 방향을 ReSpeaker LED 링에 점등")
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
            ring.set_brightness(10)
    fs = FS_DEFAULT
    win_n = int(args.window * fs)
    hop_n = max(1, int(args.hop * fs))
    rolling = np.zeros((win_n, 4), dtype="float32")  # 최근 window 만큼의 ch1~4
    print(f"ReSpeaker device={dev} | window={args.window}s hop={args.hop}s "
          f"(~{1/args.hop:.0f}Hz) | LED={'on' if ring else 'off'} | Ctrl+C 종료\n")

    multi_count = 0
    try:
        # 연속 스트림: 끊김 없이 hop 단위로 읽고, 최근 window 를 분석 (슬라이딩 윈도우)
        with sd.InputStream(samplerate=fs, channels=6, device=dev,
                            dtype="float32", blocksize=hop_n) as stream:
            while True:
                block, overflowed = stream.read(hop_n)        # ~hop 초 만큼 블록
                rolling = np.roll(rolling, -hop_n, axis=0)
                rolling[-hop_n:] = np.asarray(block)[:, 1:5]   # ch1~4 만

                rms = float(np.sqrt(np.mean(rolling ** 2)))
                if rms < args.threshold:
                    if ring:
                        ring.off()
                    print(f"\r{_compass_bar([])}  조용함 (rms={rms:.4f})        ", end="", flush=True)
                    continue

                results = estimate_multiple_directions(
                    rolling, fs=fs, num_src=args.num, algo=args.algo,
                    height_ratio=args.height, min_sep_deg=args.min_sep,
                )
                angles = [a for a, _ in results]
                if ring:
                    ring.show_directions(angles)
                label = "  ".join(f"{a:3.0f}°({d.value})" for a, d in results) or "(peak 없음)"
                lag = " ⚠느림" if overflowed else ""
                line = f"{_compass_bar(angles)}  {label}  rms={rms:.3f}{lag}"

                if len(angles) >= 2:
                    # 다중 감지 → 영구 로그 한 줄로 남김 (newline)
                    multi_count += 1
                    print(f"\r#{multi_count:<3} ★다중({len(angles)})  {line}        ")
                else:
                    # 단일/없음 → 실시간 한 줄 덮어쓰기
                    print(f"\r{line}        ", end="", flush=True)
    except KeyboardInterrupt:
        if ring:
            ring.off()
            ring.close()
        print("\n종료")


if __name__ == "__main__":
    main()
