"""doa.led_ring 단위 테스트 (순수 로직만, 하드웨어 불필요).

각도→LED 매핑과 프로토콜 payload 변환을 검증한다.
실제 USB 전송(LedRing.show 등)은 하드웨어 의존이라 다루지 않는다.
"""

import pytest

from doa.led_ring import (
    NUM_LEDS,
    angle_to_led,
    buffer_to_payload,
    directions_to_buffer,
)


def test_angle_to_led_cardinals():
    assert angle_to_led(0, offset=0) == 0
    assert angle_to_led(90, offset=0) == 3     # 90/30
    assert angle_to_led(180, offset=0) == 6
    assert angle_to_led(270, offset=0) == 9
    assert angle_to_led(360, offset=0) == 0    # wrap


def test_angle_to_led_rounds_to_nearest():
    assert angle_to_led(44, offset=0) == 1     # 44/30=1.47 → 1
    assert angle_to_led(46, offset=0) == 2     # 46/30=1.53 → 2


def test_angle_to_led_offset_wraps():
    assert angle_to_led(0, offset=3) == 3
    assert angle_to_led(0, offset=-1) == NUM_LEDS - 1


def test_directions_to_buffer_lights_correct_leds():
    buf = directions_to_buffer([0, 90])
    assert len(buf) == NUM_LEDS
    assert buf[0] != (0, 0, 0)   # 0° → LED 0 점등
    assert buf[3] != (0, 0, 0)   # 90° → LED 3 점등
    assert buf[6] == (0, 0, 0)   # 점등 안 한 LED 는 꺼짐


def test_directions_use_distinct_palette_colors():
    """서로 다른 음원은 서로 다른 색으로 구분된다."""
    buf = directions_to_buffer([0, 90])
    assert buf[0] != buf[3]


def test_buffer_to_payload_format():
    payload = buffer_to_payload([(0, 0, 0)] * NUM_LEDS)
    assert len(payload) == NUM_LEDS * 4          # [r,g,b,0]×12 = 48
    assert payload[3::4] == [0] * NUM_LEDS        # 매 4번째 바이트는 패딩 0


def test_buffer_to_payload_values_and_clamp():
    payload = buffer_to_payload([(255, 128, 1)] + [(0, 0, 0)] * (NUM_LEDS - 1))
    assert payload[:4] == [255, 128, 1, 0]


def test_buffer_to_payload_wrong_length_raises():
    with pytest.raises(ValueError):
        buffer_to_payload([(0, 0, 0)] * 11)
