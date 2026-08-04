"""
Jetson → 폰 BLE 통신 '약속'(프로토콜).

전달 경로: Jetson ─BLE→ 폰 앱(GATT 서버) → 알림 게시 → 워치가 알림 미러링으로 진동.
워치에 직접 연결하지 않는다. 워치는 폰의 진동 파형이 아니라 '알림'을 미러링하므로
방향 리듬·세기는 폰에서만 재현된다.

★ 이 파일의 UUID·바이트 포맷은 **폰쪽 AlertProtocol.kt 와 반드시 동일**해야 한다.
   (phone-app: app/src/main/java/com/example/emergencyphone/ble/AlertProtocol.kt)
   watch-app(Wear OS)은 현재 미사용 — Wear OS 워치를 쓰게 되면 같은 포맷으로 되살린다.

페이로드(4바이트):
  byte[0] sound  : 0=일반, 1=사이렌, 2=경적
  byte[1] dir    : 0=전방, 1=후방, 2=좌, 3=우, 0xFF=미상
  byte[2] motion : 0=접근, 1=멀어짐, 2=유지, 0xFF=미상
  byte[3] conf   : 신뢰도 0~100 (퍼센트)
"""

from __future__ import annotations

from core.types import Direction, Motion

# 폰 앱과 동일한 고유 식별자. 한 글자라도 다르면 서로 못 알아본다.
SERVICE_UUID = "e7a10000-2c5f-4b9a-8d3e-1f0a9b8c7d60"
ALERT_CHAR_UUID = "e7a10001-2c5f-4b9a-8d3e-1f0a9b8c7d60"
# 광고 이름 폴백용. phone-app AlertProtocol.kt 의 DEVICE_NAME 과 같아야 한다.
# (이전 값 "EmergencyWatch" 는 폰과 불일치라 이름 매칭이 죽어 있었다. UUID 매칭은
#  동작했으므로 증상은 없었지만, 스캔응답에 UUID 가 안 실리면 못 찾는다.)
DEVICE_NAME = "EmergencyPhone"

# core.types enum → 바이트 코드 매핑 (폰 enum 의 code 와 동일)
_DIR = {
    Direction.FRONT: 0, Direction.REAR: 1, Direction.LEFT: 2,
    Direction.RIGHT: 3, Direction.UNKNOWN: 0xFF,
}
_MOTION = {
    Motion.APPROACHING: 0, Motion.RECEDING: 1,
    Motion.STEADY: 2, Motion.UNKNOWN: 0xFF,
}


# 소리 종류는 매 tick '원시 분류'(info['label'])가 아니라 **디바운스된 경보
# 상태기계**(ev.kind/level, ONSET/REMIND/CLEAR)를 따른다 — 사이렌 유지 중엔
# 안정적으로 사이렌 바이트를 보내고, 해제되면 '일반(0)'을 보내 폰이 진동을 멈춘다.
_KIND_SOUND = {"siren": 1, "horn": 2}


def encode_alert(ev, info: dict) -> bytes:
    """(AlertEvent, info) → 폰으로 보낼 4바이트.

    통합 런타임의 출력이 FusedResult 가 아니라 (AlertEvent, info dict) 이므로
    이것이 유일한 인코더다.

    info 키: direction(Direction|None), motion(Motion|None), conf(float 0~1).
    - motion 은 조건부 융합(pipeline.motion_fusion) 결과다. 속도 단계는 공개 데이터
      검증에서 기준선 이하라 페이로드에 넣지 않는다 (docs/approach/validation.md).
    - 경보 없음/해제(ev.level == 'NONE')  → sound=0(일반), 폰이 진동 정지.
    - 긴급 아님·워밍업(info 값이 None)     → direction/motion 각각 미상(0xFF).
    """
    if ev is not None and getattr(ev, "level", "NONE") != "NONE":
        sound = _KIND_SOUND.get(getattr(ev, "kind", "none"), 0)   # siren→1, horn→2
    else:
        sound = 0                                                  # 경보 없음/해제 → 일반
    direction = _DIR.get(info.get("direction"), 0xFF)
    motion = _MOTION.get(info.get("motion"), 0xFF)
    conf = max(0, min(100, int(round((info.get("conf") or 0.0) * 100))))
    return bytes([sound, direction, motion, conf])
