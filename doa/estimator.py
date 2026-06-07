"""
② 방향 추정 모듈  —— 팀원 담당 (여기는 인터페이스 스텁만).

★ 이 파일을 맡은 팀원이 채우면 됩니다.
   - 입력: 4채널 AudioChunk (ReSpeaker 원시 오디오)
   - 출력: DirectionResult (전/후/좌/우)
   - 1차 접근: ReSpeaker XVF-3000 자체 DOA 값 읽기
   - 2차 접근: 4채널 TDOA(GCC-PHAT 등) 직접 구현

분류 담당(나)은 이 함수를 '호출'만 하므로, 시그니처(입출력 형식)만
core.types 의 약속대로 유지되면 된다.
"""

from __future__ import annotations

from core.types import AudioChunk, DirectionResult, Direction


def estimate_direction(chunk: AudioChunk) -> DirectionResult:
    """4채널 오디오 -> 방향. (현재는 미구현 스텁)"""
    # TODO(팀원): ReSpeaker DOA 또는 4채널 TDOA 기반 방향 추정 구현
    return DirectionResult(direction=Direction.UNKNOWN, angle_deg=None)
