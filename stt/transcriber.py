"""
④ STT(음성→텍스트) 모듈 —— 담당: 천자민.

  - 입력: 시간에 따른 AudioChunk 흐름 (1초 모노 16kHz)
  - 출력: SpeechResult (텍스트 + 긴급 키워드)

1초 청크는 STT 엔진에 너무 짧다. 그래서 ApproachDetector 처럼 '상태를 가진'
클래스로 두고, 음성이 이어지는 동안 모았다가(발화 단위) 한 번에 인식한다.
조용한 구간은 무음 게이트(VAD)로 걸러 엔진을 아예 돌리지 않는다(비용 절약).

엔진(faster-whisper)은 _Engine 인터페이스로 추상화해, 무거운 의존성은 지연 import
하고 테스트에서는 가짜 엔진을 주입할 수 있게 했다.

> ⚠️ 엔진은 16kHz 모노 float32 를 가정한다(audio.capture 가 그렇게 만들어 준다).
"""

from __future__ import annotations

import sys
from typing import List, Optional, Protocol, Tuple

import numpy as np

from core.types import AudioChunk, SpeechResult
from .config import STTConfig
from .device import resolve_runtime
from .keywords import find_keywords


# ---------------------------------------------------------------------------
# 엔진 추상화: (samples, sample_rate) -> (text, confidence, lang)
# ---------------------------------------------------------------------------
class _Engine(Protocol):
    def transcribe(
        self, samples: np.ndarray, sample_rate: int
    ) -> Tuple[str, float, Optional[str]]:
        ...


class FasterWhisperEngine:
    """faster-whisper 래퍼. 무거운 import 는 생성 시점에만 일어난다."""

    def __init__(self, cfg: STTConfig):
        from faster_whisper import WhisperModel  # 지연 import: 설치 안 해도 모듈 로드 가능

        # "auto" 면 노트북=cpu/int8, Jetson(GPU)=cuda/float16 으로 결정.
        device, compute_type = resolve_runtime(cfg.device, cfg.compute_type)
        print(f"[stt] 엔진 로드: {cfg.model_size} on {device}/{compute_type}", file=sys.stderr)
        self._model = WhisperModel(
            cfg.model_size, device=device, compute_type=compute_type
        )
        self._language = cfg.language
        self._beam_size = cfg.beam_size

    def transcribe(
        self, samples: np.ndarray, sample_rate: int
    ) -> Tuple[str, float, Optional[str]]:
        # faster-whisper 는 16kHz 모노 float32 numpy 를 직접 받는다.
        segments, info = self._model.transcribe(
            samples, language=self._language, beam_size=self._beam_size
        )
        segs = list(segments)
        text = " ".join(s.text.strip() for s in segs).strip()
        # avg_logprob(로그확률) → exp 로 대략적인 0~1 신뢰도.
        if segs:
            conf = float(np.mean([np.exp(s.avg_logprob) for s in segs]))
            conf = max(0.0, min(1.0, conf))
        else:
            conf = 0.0
        lang = getattr(info, "language", None)
        return text, conf, lang


# ---------------------------------------------------------------------------
# 보조 함수
# ---------------------------------------------------------------------------
def _to_mono(samples: np.ndarray) -> np.ndarray:
    """다채널이면 평균내 모노로, dtype 은 float32 로 맞춘다."""
    x = np.asarray(samples)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x.astype(np.float32, copy=False)


def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


# ---------------------------------------------------------------------------
# 메인 클래스
# ---------------------------------------------------------------------------
class Transcriber:
    """연속 청크를 받아 발화 단위로 텍스트를 인식. transcribe(chunk) → SpeechResult."""

    def __init__(self, config: Optional[STTConfig] = None, engine: Optional[_Engine] = None):
        self.cfg = config or STTConfig()
        self._engine = engine          # 주입 가능(테스트). None 이면 첫 인식 때 lazy-load.
        self._buf: List[np.ndarray] = []   # 음성이 이어지는 동안 누적
        self._silence_run = 0              # 음성 이후 연속 무음 청크 수
        self._had_speech = False           # 현재 버퍼에 음성이 들어있는지

    # -- 공개 API (인터페이스 약속) ----------------------------------------
    def transcribe(self, chunk: AudioChunk) -> SpeechResult:
        x = _to_mono(chunk.samples)
        sr = chunk.sample_rate
        is_voice = _rms(x) >= self.cfg.vad_rms_threshold

        if is_voice:
            self._buf.append(x)
            self._had_speech = True
            self._silence_run = 0
            if self._buffered_seconds(sr) >= self.cfg.max_utterance_seconds:
                return self._flush(sr)         # 너무 길어지면 강제로 끊어 인식
            return SpeechResult(is_speech=True)  # 아직 누적 중(텍스트는 flush 때)

        # 무음 청크
        if self._had_speech:
            self._silence_run += 1
            if self._silence_run >= self.cfg.silence_release_chunks:
                return self._flush(sr)         # 발화 끝 → 모은 걸 인식
        return SpeechResult(is_speech=False)

    def flush(self) -> SpeechResult:
        """스트림 종료 시 남은 버퍼를 강제로 인식(있으면)."""
        return self._flush(self.cfg.sample_rate)

    # -- 내부 ---------------------------------------------------------------
    def _buffered_seconds(self, sr: int) -> float:
        n = sum(len(b) for b in self._buf)
        return n / sr if sr else 0.0

    def _reset(self) -> None:
        self._buf = []
        self._silence_run = 0
        self._had_speech = False

    def _flush(self, sr: int) -> SpeechResult:
        if not self._buf:
            self._reset()
            return SpeechResult(is_speech=False)

        audio = np.concatenate(self._buf)
        self._reset()

        if len(audio) < self.cfg.min_utterance_seconds * sr:
            return SpeechResult(is_speech=False)  # 너무 짧으면 잡음으로 보고 버림

        text, conf, lang = self._get_engine().transcribe(audio, sr)
        text = (text or "").strip()
        if not text:
            return SpeechResult(is_speech=True, text="", confidence=conf, lang=lang)

        return SpeechResult(
            is_speech=True,
            text=text,
            keywords=find_keywords(text),
            confidence=conf,
            lang=lang,
        )

    def _get_engine(self) -> _Engine:
        if self._engine is None:
            self._engine = FasterWhisperEngine(self.cfg)
        return self._engine


# ---------------------------------------------------------------------------
# 파일/배열 단발 인식용 헬퍼 (버퍼링 없이 통째로)
# ---------------------------------------------------------------------------
def transcribe_array(
    samples: np.ndarray,
    sample_rate: int = STTConfig().sample_rate,
    config: Optional[STTConfig] = None,
    engine: Optional[_Engine] = None,
) -> SpeechResult:
    """이미 잘려있지 않은 통짜 오디오(예: WAV 한 파일)를 한 번에 인식."""
    cfg = config or STTConfig()
    eng = engine or FasterWhisperEngine(cfg)
    audio = _to_mono(samples)
    text, conf, lang = eng.transcribe(audio, sample_rate)
    text = (text or "").strip()
    if not text:
        return SpeechResult(is_speech=bool(audio.size), text="", confidence=conf, lang=lang)
    return SpeechResult(
        is_speech=True, text=text, keywords=find_keywords(text), confidence=conf, lang=lang
    )
