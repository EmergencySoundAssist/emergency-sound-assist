"""DoA 단일 음원 진단 — 방향이 '반대로 튀는' 원인을 (A)/(B)로 가른다.

한 방향에서만 소리를 내며 이 스크립트를 돌리면, 매 프레임마다
  · raw 방위각(pyroomacoustics, 보정 전)
  · 차량 기준 각도(_to_vehicle_angle, 보정 후) → 4방향
  · 신뢰도 = 주엽 우월도(주엽÷반대편 반원 최대) (클수록 한 방향 뚜렷 = 믿을 만)
를 찍고, Ctrl+C 시 **원형 통계**로 판정한다.

판정:
  (A) 구경 한계(통계적 튐)  → raw 각이 넓게 흩어짐(R 낮음)·반대편 비율↑
  (B) 좌표 규약/보정 오차    → raw 각은 안정(R 높음)인데 방향이 일정하게 틀어짐

실행 (airacle 환경, repo 루트):
    python -m doa.diag --truth front
    python -m doa.diag --truth front --algo MUSIC --window 0.5

--truth 에 실제 소리 방향(front/rear/left/right)을 주면, "안정적인데 틀림"(B)을
자동으로 짚어준다. Ctrl+C 로 종료.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from doa import config, led_ring
from doa.estimator import _to_vehicle_angle, angle_to_direction
from doa.multi_source import (
    FS_DEFAULT,
    find_respeaker_device,
    spatial_spectrum,
    spectrum_confidence,
)

# 실제 음원 방향(차량 기준) → 기대 각도(도)
TRUTH_DEG = {"front": 0.0, "right": 90.0, "rear": 180.0, "left": 270.0}


def _circ_stats(angles_deg: np.ndarray):
    """원형 평균·집중도 R(0~1)·원형 표준편차(도)."""
    r = np.deg2rad(angles_deg)
    c, s = np.cos(r).mean(), np.sin(r).mean()
    R = float(np.hypot(c, s))                       # 1=완전 일관, 0=완전 분산
    mean = float(np.degrees(np.arctan2(s, c)) % 360.0)
    std = float(np.degrees(np.sqrt(-2.0 * np.log(max(R, 1e-12)))))
    return mean, R, std


def _ang_diff(a: float, b: float) -> float:
    """두 각도 최소차(0~180)."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", choices=sorted(TRUTH_DEG), default=None,
                    help="실제 소리 방향(차량 기준). 주면 (B) 자동 판정")
    ap.add_argument("--threshold", type=float, default=config.THRESHOLD,
                    help=f"이 RMS 미만이면 무음 취급 (config={config.THRESHOLD})")
    ap.add_argument("--window", type=float, default=config.WINDOW,
                    help=f"분석 창(초) (config={config.WINDOW})")
    ap.add_argument("--hop", type=float, default=config.HOP,
                    help=f"갱신 간격(초) (config={config.HOP})")
    ap.add_argument("--algo", default=config.ALGO,
                    help=f"SRP/MUSIC (config={config.ALGO})")
    ap.add_argument("--num", type=int, default=config.NUM_SRC,
                    help=f"num_src — multi_live 게이트와 같아야 conf 가 전이됨 "
                         f"(MUSIC 에서 특히 중요, config={config.NUM_SRC})")
    ap.add_argument("--led", action="store_true",
                    help="추정한 raw 방향을 LED 링에 점등 (소리 위치 vs 불 위치로 (B) 육안 확인)")
    ap.add_argument("--device", type=int, default=None,
                    help="입력 장치 인덱스 (미지정 시 자동 탐지). sd.query_devices() 로 확인")
    args = ap.parse_args()

    import sounddevice as sd

    dev = args.device if args.device is not None else find_respeaker_device(sd)
    if dev is None:
        print("ReSpeaker(6채널)를 못 찾음. sd.query_devices() 로 확인 후 --device 인덱스 지정.")
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
    rolling = np.zeros((win_n, 4), dtype="float32")

    truth_deg = TRUTH_DEG[args.truth] if args.truth else None
    print(f"단일 음원 진단 | device={dev} | window={args.window}s hop={args.hop}s "
          f"| algo={args.algo} | truth={args.truth or '미지정'} | Ctrl+C 종료\n")
    print(" raw  → veh  → 방향   신뢰도")

    raws: list[float] = []      # raw 방위각
    vehs: list[float] = []      # 차량 기준 각도
    confs: list[float] = []     # 주엽 우월도
    try:
        with sd.InputStream(samplerate=fs, channels=6, device=dev,
                            dtype="float32", blocksize=hop_n) as stream:
            while True:
                block, _ = stream.read(hop_n)
                rolling = np.roll(rolling, -hop_n, axis=0)
                rolling[-hop_n:] = np.asarray(block)[:, 1:5]

                rms = float(np.sqrt(np.mean(rolling ** 2)))
                if rms < args.threshold:
                    print(f"\r  (조용함 rms={rms:.4f})            ", end="", flush=True)
                    continue

                az, spec = spatial_spectrum(rolling, fs=fs, algo=args.algo, num_src=args.num)
                spec = np.asarray(spec, dtype=float)
                if spec.size == 0 or not np.any(spec > 0):
                    continue
                k = int(np.argmax(spec))
                raw = float(az[k])
                veh = _to_vehicle_angle(raw)
                direction = angle_to_direction(raw)
                conf = spectrum_confidence(spec)  # 주엽 우월도 (config.CONF_MIN 과 동일 지표)

                raws.append(raw)
                vehs.append(veh)
                confs.append(conf)
                if ring:
                    ring.show_directions([raw])   # raw 방향에 점등 (소리 위치와 비교)
                print(f"\r{raw:5.0f} → {veh:4.0f} → {direction.value:<5} "
                      f"conf={conf:4.1f}        ", end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        if ring:
            ring.off()
            ring.close()

    n = len(raws)
    print(f"\n\n=== 진단 요약 (유효 프레임 {n}) ===")
    if n < 5:
        print("표본이 너무 적음. 소리를 더 길게 내고 다시 시도하세요.")
        return

    raw_arr = np.asarray(raws)
    raw_mean, raw_R, raw_std = _circ_stats(raw_arr)
    veh_mean, _, _ = _circ_stats(np.asarray(vehs))
    conf_mean = float(np.mean(confs))
    # 평균에서 90° 이상 벗어난(=반대편으로 튄) 프레임 비율
    flip_rate = float(np.mean([_ang_diff(a, raw_mean) > 90.0 for a in raw_arr]))

    print(f"raw 방위각 : 평균 {raw_mean:.0f}°  집중도 R={raw_R:.2f}  "
          f"원형표준편차 {raw_std:.0f}°")
    print(f"차량 기준  : 평균 {veh_mean:.0f}°  → {angle_to_direction(raw_mean).value}")
    print(f"반대편 튐 비율 : {flip_rate*100:.0f}%   평균 신뢰도 conf={conf_mean:.1f}")

    print("\n--- 판정 ---")
    stable = raw_R >= 0.7 and flip_rate < 0.2          # 충분히 안정적인가
    if not stable:
        print("(A) 구경 한계 — raw 각이 흩어지거나 자주 반대편으로 튐.")
        print("    → 신뢰도 게이팅 + 시간 다수결(스무딩) 도입이 직접 처방.")
    if truth_deg is not None:
        err = _ang_diff(veh_mean, truth_deg)
        print(f"(B) 실제={args.truth}({truth_deg:.0f}°) vs 추정 {veh_mean:.0f}° → 오차 {err:.0f}°")
        if stable and err > 60.0:
            print("    → 안정적인데 일정하게 틀어짐 = 좌표 규약/보정 오차.")
            print(f"    → 약 {round(err/30)*30}° 회전 또는 좌우반전 의심. "
                  "config의 REAR_RAW_DEG / MIRROR 조정으로 상수 픽스 가능.")
        elif stable and err <= 60.0:
            print("    → 안정적이고 방향도 대체로 맞음. 보정 OK.")
    elif stable:
        print("(B) raw 각은 안정적. --truth 를 주면 방향 정오(보정 여부)까지 자동 판정.")


if __name__ == "__main__":
    main()
