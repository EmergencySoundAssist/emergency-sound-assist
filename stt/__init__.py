"""
④ STT(음성→텍스트) 모듈.

주변 음성(경찰 확성기·외침·안내방송 등)을 텍스트로 바꾸고,
운전 관련 긴급 키워드(구급차/비키세요/정지 등)를 짚어 청각장애 운전자에게 보여 준다.

공개 API:
    Transcriber.transcribe(chunk) -> SpeechResult   # 연속 청크(실시간) 권장
    transcribe_array(samples)     -> SpeechResult   # 파일 한 방에 인식

자세한 설계 → docs/stt/design.md
"""

from .config import STTConfig, EMERGENCY_KEYWORDS
from .device import resolve_runtime, cuda_available
from .keywords import find_keywords
from .transcriber import Transcriber, FasterWhisperEngine, transcribe_array

__all__ = [
    "STTConfig",
    "EMERGENCY_KEYWORDS",
    "find_keywords",
    "resolve_runtime",
    "cuda_available",
    "Transcriber",
    "FasterWhisperEngine",
    "transcribe_array",
]
