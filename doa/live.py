"""① 자체 DoA 경량 실시간 데모 (XVF-3000 방향 레지스터 폴링).

ReSpeaker 자체 DoA 값을 0.2초마다 읽어 4방향으로 출력한다. 오디오 캡처·pyroomacoustics
없이 **numpy + pyusb** 만으로 동작 — Jetson 경량 경로/최초 브링업용. (다중 음원·LED·스무딩은
`python -m doa.multi_live`.)

실행: 레포 루트에서 `python -m doa.live` (또는 설치 후 `doa-live`). Ctrl+C 로 종료.
"""

import time

from doa.estimator import estimate_direction


def main():
    print("실시간 방향 감지 시작 (Ctrl+C로 종료)\n")
    try:
        while True:
            result = estimate_direction(None)
            if result.angle_deg is None:
                # 장치 미연결 / pyusb·libusb 없음 / USB 권한(udev) — 크래시 대신 안내
                print("\r장치 없음 — ReSpeaker 미연결·pyusb/libusb 미설치·USB 권한(udev) 확인        ",
                      end="", flush=True)
            else:
                print(f"\r{result.angle_deg:>5.0f}° → {result.direction.value}        ",
                      end="", flush=True)
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n종료")


if __name__ == "__main__":
    main()
