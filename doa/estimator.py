"""
② 방향 추정 모듈.

노트북 단계: ReSpeaker 실물이 없으므로 각도→4방향 변환 로직만 동작한다.
Jetson 단계: ReSpeaker `Tuning.direction`(0~359°)을 폴링해 `angle_deg`에 채운다.

- 입력: 4채널 AudioChunk (ReSpeaker 원시 오디오)  ※ 1단계에선 사용 안 함
- 출력: DirectionResult (전/후/좌/우/unknown)
"""

from __future__ import annotations

from typing import Optional

from core.types import AudioChunk, Direction, DirectionResult

# ---------------------------------------------------------------------------
# ReSpeaker 의존성 (있으면 쓰고, 없으면 조용히 비활성화)
# ---------------------------------------------------------------------------
# pyusb 와 Seeed 의 tuning.py 가 모두 있어야 ReSpeaker DoA 폴링이 가능.
# 노트북엔 둘 다 없을 수 있으므로 import 실패해도 모듈 로드는 깨지지 않게 둔다.
try:
    import usb.core  # type: ignore
    from .respeaker_tuning import Tuning  # type: ignore
    _RESPEAKER_LIB_AVAILABLE = True
except ImportError:
    _RESPEAKER_LIB_AVAILABLE = False


def angle_to_direction(angle_deg: float) -> Direction:
    """0~359° → 4방향. 경계는 [start, end) 반개구간으로 처리.

    전 315~45° / 우 45~135° / 후 135~225° / 좌 225~315°
    (docs/doa/design.md 의 1단계 매핑 규칙)
    """
    deg = angle_deg % 360.0
    if 45.0 <= deg < 135.0:
        return Direction.RIGHT
    if 135.0 <= deg < 225.0:
        return Direction.REAR
    if 225.0 <= deg < 315.0:
        return Direction.LEFT
    return Direction.FRONT  # 315~360 또는 0~45


def _read_respeaker_angle() -> Optional[float]:
    """ReSpeaker XVF-3000 의 자체 DoA 값(0~359°)을 읽는다.

    라이브러리 또는 장치가 없으면 None 을 반환한다(노트북 단계 정상 경로).
    하드웨어 도착 후 아래 TODO 블록만 채우면 1단계 MVP 완성.
    """
    if not _RESPEAKER_LIB_AVAILABLE:
        return None
    # TODO(하드웨어 도착 후):
    #   1) dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    #      - 실패 시 None 반환
    #   2) mic = Tuning(dev)
    #   3) return float(mic.direction)
    return None


def estimate_direction(
    chunk: AudioChunk,
    angle_deg: Optional[float] = None,
) -> DirectionResult:
    """4채널 오디오 → 방향.

    우선순위:
        1) 호출 측에서 `angle_deg` 를 직접 주면 그 값을 사용 (테스트/주입용)
        2) 아니면 ReSpeaker 에서 폴링 시도
        3) 둘 다 실패하면 UNKNOWN

    `chunk` 자체는 1단계(MVP)에선 사용하지 않으며, 2단계 GCC-PHAT 에서 활용 예정.
    """
    if angle_deg is None:
        angle_deg = _read_respeaker_angle()
    if angle_deg is None:
        return DirectionResult(direction=Direction.UNKNOWN, angle_deg=None)
    return DirectionResult(direction=angle_to_direction(angle_deg), angle_deg=angle_deg)

if __name__ == "__main__":
    for a in [0, 90, 180, 270, 44, 45, 359]:
        print(a, "→", angle_to_direction(a))
    print("주입 없음 →", estimate_direction(None))   # UNKNOWN 나와야 정상