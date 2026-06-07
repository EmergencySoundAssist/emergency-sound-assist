"""
③ 접근/멀어짐 판단 모듈  —— 팀원 담당 (여기는 인터페이스 스텁만).

★ 이 파일을 맡은 팀원이 채우면 됩니다.
   - 입력: 시간에 따른 AudioChunk 흐름 (사이렌일 때만 의미 있음)
   - 출력: ApproachResult (접근/멀어짐/유지)
   - 핵심 단서: 도플러 효과(주파수 변화) + 음량(에너지) 변화
       · 주파수 ↑ & 음량 ↑  -> 접근
       · 주파수 ↓ & 음량 ↓  -> 멀어짐

상태(이전 청크들)를 기억해야 하므로 클래스로 두는 편이 자연스럽다.
"""

from __future__ import annotations

from core.types import AudioChunk, ApproachResult, Motion


class ApproachDetector:
    """연속된 청크를 받아 접근/멀어짐을 판단(현재는 미구현 스텁)."""

    def update(self, chunk: AudioChunk) -> ApproachResult:
        # TODO(팀원): 도플러(주파수 추세) + 음량 추세 분석으로 motion 판단
        return ApproachResult(motion=Motion.UNKNOWN)
