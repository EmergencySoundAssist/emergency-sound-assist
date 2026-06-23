"""ReSpeaker USB Mic Array — 12 RGB LED 링 제어.

공식 LED 제어 프로토콜을 pyusb 로 직접 구현 (별도 pixel_ring 설치 불필요).
  전송: ctrl_transfer(CTRL_OUT|VENDOR|DEVICE, bRequest=0, wValue=cmd, wIndex=0x1C, data)
  · cmd=6    : 커스텀 모드 — data = [r, g, b, 0] × 12 (LED 12개 각각 색)
  · cmd=0x20 : 밝기      — data = [brightness]
출처: https://github.com/respeaker/pixel_ring (wiki: ReSpeaker USB 4 Mic Array LED Control Protocol)

LED 제어는 USB '제어 인터페이스'(VID 0x2886 / PID 0x0018)를 사용하며,
오디오 캡처(USB Audio)와 별개 인터페이스라 동시에 쓸 수 있다.
방향추정 결과(여러 각도)를 해당 LED에 점등해 "소리가 어디서 오는지"를 표시하는 용도.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

NUM_LEDS = 12
DEG_PER_LED = 360.0 / NUM_LEDS

VID, PID = 0x2886, 0x0018
_TIMEOUT = 8000
_WINDEX = 0x1C
_CMD_SHOW = 6
_CMD_BRIGHTNESS = 0x20

# LED 인덱스 보정: 방위각 0°가 물리적으로 어느 LED인지 (실물 보고 조정).
# 방향추정 각도 규약과 LED 배치가 어긋나면 이 값으로 회전 보정한다.
LED_OFFSET = 0

RGB = Tuple[int, int, int]

# 동시 여러 음원을 구분해 보여줄 색 팔레트 (음원 1·2·3 …)
PALETTE: Tuple[RGB, ...] = (
    (255, 0, 0),    # 빨강
    (0, 0, 255),    # 파랑
    (0, 255, 0),    # 초록
    (255, 255, 0),  # 노랑
)


def angle_to_led(angle_deg: float, offset: int = LED_OFFSET) -> int:
    """방위각(0~360°) → LED 인덱스(0~11)."""
    return int(round(angle_deg / DEG_PER_LED) + offset) % NUM_LEDS


def directions_to_buffer(
    angles_deg: Sequence[float],
    colors: Optional[Sequence[RGB]] = None,
    offset: int = LED_OFFSET,
) -> List[RGB]:
    """방향 각도들 → 12-LED 색 버퍼. 각 방향을 팔레트 색으로 점등.

    colors 미지정 시 PALETTE 를 순서대로 사용한다. 같은 LED 에 겹치면 나중 것이 덮어쓴다.
    """
    buf: List[RGB] = [(0, 0, 0)] * NUM_LEDS
    pal = colors if colors is not None else PALETTE
    for i, a in enumerate(angles_deg):
        buf[angle_to_led(a, offset)] = pal[i % len(pal)]
    return buf


def buffer_to_payload(buf: Sequence[RGB]) -> List[int]:
    """12-LED 색 버퍼 → 프로토콜 payload [r,g,b,0]×12 (48바이트)."""
    if len(buf) != NUM_LEDS:
        raise ValueError(f"LED 버퍼 길이는 {NUM_LEDS} 여야 함. 받은 길이={len(buf)}")
    data: List[int] = []
    for (r, g, b) in buf:
        data += [int(r) & 0xFF, int(g) & 0xFF, int(b) & 0xFF, 0]
    return data


class LedRing:
    """ReSpeaker LED 링. find() 로 생성한다."""

    def __init__(self, dev):
        self.dev = dev

    def _write(self, cmd: int, data: List[int]) -> None:
        import usb.util
        self.dev.ctrl_transfer(
            usb.util.CTRL_OUT | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
            0, cmd, _WINDEX, data, _TIMEOUT,
        )

    def set_brightness(self, brightness: int) -> None:
        """밝기 0~31 (0x00~0x1F)."""
        self._write(_CMD_BRIGHTNESS, [int(brightness) & 0xFF])

    def show(self, buf: Sequence[RGB]) -> None:
        """12개 (r,g,b) 버퍼를 LED 에 표시 (커스텀 모드)."""
        self._write(_CMD_SHOW, buffer_to_payload(buf))

    def show_directions(
        self, angles_deg: Sequence[float], offset: int = LED_OFFSET
    ) -> None:
        """방향 각도들을 바로 LED 에 점등."""
        self.show(directions_to_buffer(angles_deg, offset=offset))

    def off(self) -> None:
        self.show([(0, 0, 0)] * NUM_LEDS)

    def close(self) -> None:
        import usb.util
        usb.util.dispose_resources(self.dev)


def find() -> Optional[LedRing]:
    """ReSpeaker LED 링 장치를 찾는다. 없으면 None (pyusb 필요)."""
    import usb.core
    dev = usb.core.find(idVendor=VID, idProduct=PID)
    return LedRing(dev) if dev else None


if __name__ == "__main__":
    # 자가 점검: LED 가 동작하는지 한 바퀴 돌려본다 (오디오 무관, 하드웨어 필요).
    import time

    ring = find()
    if ring is None:
        raise SystemExit("ReSpeaker(LED 링)를 찾지 못함 — USB 연결/pyusb 확인")
    ring.set_brightness(10)
    print("LED 한 바퀴 점등 테스트 (Ctrl+C 종료)")
    try:
        i = 0
        while True:
            ring.show_directions([i * DEG_PER_LED])  # 한 칸씩 회전
            i = (i + 1) % NUM_LEDS
            time.sleep(0.1)
    except KeyboardInterrupt:
        ring.off()
        ring.close()
        print("\n종료")
