"""
② 방향 추정 모듈.

노트북 단계: ReSpeaker 실물이 없으므로 각도→4방향 변환 로직만 구현한다.
실제 각도는 Jetson 단계에서 ReSpeaker `Tuning.direction`(0~359°)으로 채울 예정.

- 입력: 4채널 AudioChunk (ReSpeaker 원시 오디오)  ※ 현재는 사용 안 함
- 출력: DirectionResult (전/후/좌/우/unknown)
"""

from __future__ import annotations

from typing import Optional

from core.types import AudioChunk, Direction, DirectionResult


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


def estimate_direction(
    chunk: AudioChunk,
    angle_deg: Optional[float] = None,
) -> DirectionResult:
    """4채널 오디오 → 방향.

    노트북 단계에서는 `angle_deg`를 직접 주입해 인터페이스 테스트.
    Jetson 단계에서 ReSpeaker `Tuning.direction` 폴링 값으로 교체 예정.
    """
    if angle_deg is None:
        return DirectionResult(direction=Direction.UNKNOWN, angle_deg=None)
    return DirectionResult(direction=angle_to_direction(angle_deg), angle_deg=angle_deg)