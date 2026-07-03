"""
Jetson → 워치 BLE 통신 '약속'(프로토콜).

★ 이 파일의 UUID·바이트 포맷은 **워치쪽 AlertProtocol.kt 와 반드시 동일**해야 한다.
   (watch-app: app/src/main/java/com/example/emergencywatch/ble/AlertProtocol.kt)

페이로드(4바이트):
  byte[0] sound  : 0=일반, 1=사이렌, 2=경적
  byte[1] dir    : 0=전방, 1=후방, 2=좌, 3=우, 0xFF=미상
  byte[2] motion : 0=접근, 1=멀어짐, 2=유지, 0xFF=미상
  byte[3] conf   : 신뢰도 0~100 (퍼센트)
"""

from __future__ import annotations

from core.types import FusedResult, SoundClass, Direction, Motion

# 워치와 동일한 고유 식별자. 한 글자라도 다르면 서로 못 알아본다.
SERVICE_UUID = "e7a10000-2c5f-4b9a-8d3e-1f0a9b8c7d60"
ALERT_CHAR_UUID = "e7a10001-2c5f-4b9a-8d3e-1f0a9b8c7d60"
DEVICE_NAME = "EmergencyWatch"

# core.types enum → 바이트 코드 매핑 (워치 enum 의 code 와 동일)
_SOUND = {SoundClass.NORMAL_TRAFFIC: 0, SoundClass.SIREN: 1, SoundClass.HORN: 2}
_DIR = {
    Direction.FRONT: 0, Direction.REAR: 1, Direction.LEFT: 2,
    Direction.RIGHT: 3, Direction.UNKNOWN: 0xFF,
}
_MOTION = {
    Motion.APPROACHING: 0, Motion.RECEDING: 1,
    Motion.STEADY: 2, Motion.UNKNOWN: 0xFF,
}


def encode(fused: FusedResult) -> bytes:
    """FusedResult → 워치로 보낼 4바이트."""
    sound = _SOUND.get(fused.sound.label, 0)
    direction = _DIR.get(fused.direction.direction, 0xFF)
    motion = _MOTION.get(fused.approach.motion, 0xFF)
    conf = max(0, min(100, int(round(fused.sound.confidence * 100))))
    return bytes([sound, direction, motion, conf])
