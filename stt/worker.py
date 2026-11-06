"""
STT 백그라운드 워커 — 메인 파이프라인을 막지 않고 뒤에서 음성 인식.

문제: faster-whisper 인식은 수 초 블로킹이라, 메인 루프에서 직접 부르면 그동안
      분류·방향·접근이 멈춘다 → STT 인식 중 사이렌을 놓칠 수 있다(위험).
해결: 인식을 별도 스레드로 돌린다. 메인은 feed()/reset() 만 호출하고 즉시 돌아간다.

사용:
    worker = STTWorker(transcriber); worker.start()
    worker.feed(chunk)       # 평상시 청크 전달(블로킹 X). 큐가 차면 버린다(실시간 우선).
    worker.reset()           # 긴급 진입 → 발화 버퍼 비움(우선순위 전환)
    cap = worker.latest()    # 완성된 자막을 한 번 꺼냄(소비). 없으면 None
    worker.stop()            # 종료
"""

from __future__ import annotations

import queue
import sys
import threading
from dataclasses import asdict, dataclass
from typing import Optional

from core.types import AudioChunk, SpeechResult


@dataclass(frozen=True)
class STTWorkerStatus:
    alive: bool
    generation: int
    queued: int
    dropped_chunks: int
    reset_count: int
    last_error: Optional[str]


class STTWorker:
    """Transcriber 를 백그라운드 스레드에서 돌리는 래퍼. 인터페이스: feed/reset/latest/stop."""

    def __init__(self, transcriber, max_queue: int = 8):
        self._t = transcriber
        self._q: "queue.Queue" = queue.Queue(maxsize=max_queue)
        self._latest: Optional[SpeechResult] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._generation = 0                    # reset마다 증가 → 이전 작업 결과 무효화
        self._applied_generation = 0            # Transcriber.reset까지 반영된 세대(워커 전용)
        self._dropped_chunks = 0
        self._reset_count = 0
        self._last_error: Optional[str] = None
        self._thread = threading.Thread(target=self._run, name="stt-worker", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def feed(self, chunk: AudioChunk) -> None:
        """평상시 청크를 워커에 넘긴다(즉시 반환). 큐가 차 있으면 버린다 — 실시간 우선."""
        with self._lock:
            generation = self._generation
        try:
            self._q.put_nowait(("chunk", generation, chunk))
        except queue.Full:
            with self._lock:
                self._dropped_chunks += 1

    def reset(self) -> None:
        """긴급 진입 → 이전 세대의 대기·진행·완료 자막을 모두 무효화한다."""
        with self._lock:
            self._generation += 1
            generation = self._generation
            self._latest = None
            self._reset_count += 1
        self._discard_pending_chunks()
        # 워커가 queue.get()에서 기다리는 중이면 즉시 깨운다. 가득 차도 0.2초 timeout으로 깨어난다.
        try:
            self._q.put_nowait(("wake", generation, None))
        except queue.Full:
            pass

    def latest(self) -> Optional[SpeechResult]:
        """워커가 완성한 자막을 한 번 꺼낸다(꺼내면 비워짐). 없으면 None."""
        with self._lock:
            r, self._latest = self._latest, None
            return r

    def status(self) -> dict:
        """대시보드/진단용 워커 상태. 읽기만 하며 블로킹하지 않는다."""
        with self._lock:
            status = STTWorkerStatus(
                alive=self._thread.is_alive(),
                generation=self._generation,
                queued=self._q.qsize(),
                dropped_chunks=self._dropped_chunks,
                reset_count=self._reset_count,
                last_error=self._last_error,
            )
        return asdict(status)

    def stop(self, timeout: float = 2.0) -> None:
        """이전 결과를 무효화하고 워커 종료를 요청한 뒤 제한시간만큼 기다린다."""
        with self._lock:
            self._generation += 1
            generation = self._generation
            self._latest = None
        self._stop.set()
        self._discard_pending_chunks()
        try:
            self._q.put_nowait(("stop", generation, None))
        except queue.Full:
            pass
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=max(0.0, timeout))

    def _discard_pending_chunks(self) -> None:
        """아직 시작하지 않은 음성 청크/제어 토큰을 제거한다."""
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                return

    def _apply_reset_if_needed(self) -> None:
        with self._lock:
            generation = self._generation
        if generation != self._applied_generation:
            self._t.reset()
            self._applied_generation = generation

    # ------------------------------------------------------------------
    def _run(self) -> None:
        """워커 루프: 큐에서 명령을 받아 처리. transcribe 블로킹은 이 스레드 안에서만."""
        while not self._stop.is_set():
            self._apply_reset_if_needed()
            try:
                kind, generation, data = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            if kind == "stop":
                break
            if kind == "wake":
                continue
            with self._lock:
                current = self._generation
            if generation != current:             # reset 전에 대기하던 청크
                continue
            try:
                res = self._t.transcribe(data)    # ← 블로킹(메인 아닌 이 스레드에서)
            except Exception as exc:              # 워커를 죽이지 말고 상태로 노출
                message = f"{type(exc).__name__}: {exc}"
                with self._lock:
                    if generation == self._generation:
                        self._last_error = message
                print(f"[stt] ⚠️ 인식 실패(워커 유지): {message}", file=sys.stderr)
                continue
            with self._lock:
                # 변환 도중 긴급 reset/stop이 발생했으면 완성된 이전 자막을 폐기한다.
                if generation != self._generation or self._stop.is_set():
                    continue
                self._last_error = None
                if res.is_speech and res.text:
                    self._latest = res
