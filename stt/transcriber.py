"""
④ STT(음성→텍스트) 모듈 —— 담당: 천자민.

  - 입력: 시간에 따른 AudioChunk 흐름 (1초 모노 16kHz)
  - 출력: SpeechResult (텍스트)

1초 청크는 STT 엔진에 너무 짧다. 그래서 ApproachDetector 처럼 '상태를 가진'
클래스로 두고, 음성이 이어지는 동안 모았다가(발화 단위) 한 번에 인식한다.
조용한 구간은 무음 게이트(VAD)로 걸러 엔진을 아예 돌리지 않는다(비용 절약).

엔진(faster-whisper)은 _Engine 인터페이스로 추상화해, 무거운 의존성은 지연 import
하고 테스트에서는 가짜 엔진을 주입할 수 있게 했다.

> ⚠️ 엔진은 16kHz 모노 float32 를 가정한다(audio.capture 가 그렇게 만들어 준다).
"""

from __future__ import annotations

import sys
from typing import Callable, List, Optional, Protocol, Tuple

import numpy as np

from core.types import AudioChunk, SpeechResult
from .config import STTConfig
from .device import resolve_runtime
from .vad import make_vad


# ---------------------------------------------------------------------------
# 엔진 추상화: (samples, sample_rate) -> (text, confidence, lang)
# ---------------------------------------------------------------------------
class _Engine(Protocol):
    def transcribe(
        self, samples: np.ndarray, sample_rate: int
    ) -> Tuple[str, float, Optional[str]]:
        ...


# CUDA 메모리 부족/관련 에러로 보이는 메시지 조각 (대소문자 무시).
_OOM_HINTS = (
    "out of memory", "illegal memory", "cublas", "cudnn", "alloc",
    "cuda failed", "cuda error", "device memory", "oom",
)


def _looks_like_oom(exc: Exception) -> bool:
    """예외가 (CUDA) 메모리 부족 계열인지 대략 판별."""
    msg = str(exc).lower()
    return any(h in msg for h in _OOM_HINTS)


def _fallback_chain(device: str, compute_type: str):
    """정확도 high → 안전 순으로 (device, compute_type) 폴백 체인.

    Orin Nano 통합 8GB 가 꽉 차면(다른 AI 모델 동시 구동) float16 부터 차례로 내려간다:
      cuda/float16 → cuda/int8_float16 → cuda/int8 → cpu/int8(최후 안전망)
    """
    if device != "cuda":
        return [(device, compute_type)]
    order = ["float16", "int8_float16", "int8"]
    if compute_type in order:
        chain = [("cuda", ct) for ct in order[order.index(compute_type):]]
    else:
        chain = [("cuda", compute_type)] + [("cuda", ct) for ct in order]
    chain.append(("cpu", "int8"))     # GPU 가 끝내 모자라면 CPU 로(느려도 안 죽음)
    return chain


class FasterWhisperEngine:
    """faster-whisper 래퍼. 무거운 import 는 생성 시점에만 일어난다.

    메모리 부족(OOM/Illegal Memory Access)이 나면 더 가벼운 정밀도로 자동 폴백한다.
    로드 시점과 변환 시점 모두 대응.
    """

    def __init__(self, cfg: STTConfig, model_factory=None):
        self._cfg = cfg
        if model_factory is None:                 # 지연 import(설치 안 해도 모듈 로드 가능)
            from faster_whisper import WhisperModel
            model_factory = WhisperModel
        self._factory = model_factory

        device, compute_type = resolve_runtime(cfg.device, cfg.compute_type)
        self._chain = _fallback_chain(device, compute_type)
        self._ci = 0                              # 현재 폴백 단계
        self._model = None
        self._device = device
        self._compute_type = compute_type

        self._language = cfg.language
        self._beam_size = cfg.beam_size
        self._no_speech_threshold = cfg.no_speech_threshold
        self._log_prob_threshold = cfg.log_prob_threshold
        self._compression_ratio_threshold = cfg.compression_ratio_threshold

        self._build_with_fallback()

    # -- 모델 로드 (OOM 폴백) ----------------------------------------------
    def _build_with_fallback(self) -> None:
        while True:
            device, ct = self._chain[self._ci]
            try:
                print(f"[stt] 엔진 로드: {self._cfg.model_size} on {device}/{ct}"
                      f" (cpu_threads={self._cfg.cpu_threads})", file=sys.stderr)
                self._model = self._factory(
                    self._cfg.model_size, device=device, compute_type=ct,
                    cpu_threads=self._cfg.cpu_threads, num_workers=self._cfg.num_workers,
                )
                self._device, self._compute_type = device, ct
                return
            except Exception as e:                # noqa: BLE001 — 폴백 판단 후 재-raise
                if not _looks_like_oom(e) or self._ci + 1 >= len(self._chain):
                    raise
                nd, nc = self._chain[self._ci + 1]
                print(f"[stt] ⚠️ 메모리 부족({device}/{ct}) → {nd}/{nc} 폴백: {e}",
                      file=sys.stderr)
                self._ci += 1

    def _advance_or_raise(self, exc: Exception) -> None:
        """변환 중 OOM → 다음 단계로 재빌드. 더 내려갈 곳 없으면 그대로 raise."""
        if not _looks_like_oom(exc) or self._ci + 1 >= len(self._chain):
            raise exc
        nd, nc = self._chain[self._ci + 1]
        print(f"[stt] ⚠️ 변환 중 메모리 부족({self._device}/{self._compute_type}) "
              f"→ {nd}/{nc} 폴백", file=sys.stderr)
        self._ci += 1
        self._build_with_fallback()

    # -- 인식 (변환 시점 OOM 폴백) -----------------------------------------
    def transcribe(
        self, samples: np.ndarray, sample_rate: int
    ) -> Tuple[str, float, Optional[str]]:
        while True:
            try:
                # faster-whisper 는 16kHz 모노 float32 numpy 를 직접 받는다.
                # condition_on_previous_text=False: 발화 단위 독립 인식(반복/환각↓).
                segments, info = self._model.transcribe(
                    samples, language=self._language, beam_size=self._beam_size,
                    condition_on_previous_text=False,
                    # 환각 가드(기본값 명시). 끄면 VAD 넓힌 만큼 노이즈→헛인식이 샌다.
                    no_speech_threshold=self._no_speech_threshold,
                    log_prob_threshold=self._log_prob_threshold,
                    compression_ratio_threshold=self._compression_ratio_threshold,
                )
                segs = list(segments)
            except Exception as e:                # noqa: BLE001
                self._advance_or_raise(e)         # OOM 이면 더 가벼운 정밀도로 재빌드 후 재시도
                continue
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


def _normalize_rms(x: np.ndarray, target_rms: float, max_gain: float) -> np.ndarray:
    """먼/조용한 음성을 목표 RMS 로 키운다(범위↑). 순수 노이즈 폭주는 max_gain 으로 제한.

    이미 큰 신호는 증폭하지 않고(게인 1 상한은 아님 — 줄일 수도 있음), 클리핑은 [-1,1] 로 막는다.
    """
    rms = _rms(x)
    if rms <= 1e-6:
        return x
    gain = min(target_rms / rms, max_gain)
    return np.clip(x * gain, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# 메인 클래스
# ---------------------------------------------------------------------------
class Transcriber:
    """연속 청크를 받아 발화 단위로 텍스트를 인식. transcribe(chunk) → SpeechResult."""

    def __init__(self, config: Optional[STTConfig] = None, engine: Optional[_Engine] = None,
                 on_status: Optional[Callable[[str], None]] = None):
        self.cfg = config or STTConfig()
        self._engine = engine          # 주입 가능(테스트). None 이면 첫 인식 때 lazy-load.
        self._on_status = on_status    # 상태 콜백("transcribing" 등) — UI 표시용
        self._vad = make_vad(self.cfg)     # webrtcvad(있으면) / energy 폴백
        self._buf: List[np.ndarray] = []   # 음성이 이어지는 동안 누적
        self._silence_run = 0              # 음성 이후 연속 무음 청크 수
        self._had_speech = False           # 현재 버퍼에 음성이 들어있는지

    def _emit(self, status: str) -> None:
        if self._on_status:
            self._on_status(status)

    # -- 공개 API (인터페이스 약속) ----------------------------------------
    def transcribe(self, chunk: AudioChunk) -> SpeechResult:
        x = _to_mono(chunk.samples)
        sr = chunk.sample_rate
        is_voice = self._vad.is_speech(x, sr)

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

        if self.cfg.normalize_audio:              # 먼/조용한 음성 살리기(범위↑)
            audio = _normalize_rms(audio, self.cfg.target_rms, self.cfg.max_gain)

        self._emit("transcribing")                # UI: '변환 중' 표시 (엔진 호출은 블로킹)
        text, conf, lang = self._get_engine().transcribe(audio, sr)
        text = (text or "").strip()
        return SpeechResult(is_speech=True, text=text, confidence=conf, lang=lang)

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
    return SpeechResult(is_speech=bool(audio.size), text=text, confidence=conf, lang=lang)
