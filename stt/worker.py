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
import threading
from typing import Optional

from core.types import AudioChunk, SpeechResult


class STTWorker:
    """Transcriber 를 백그라운드 스레드에서 돌리는 래퍼. 인터페이스: feed/reset/latest/stop."""

    def __init__(self, transcriber, max_queue: int = 8):
        self._t = transcriber
        self._q: "queue.Queue" = queue.Queue(maxsize=max_queue)
        self._latest: Optional[SpeechResult] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="stt-worker", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def feed(self, chunk: AudioChunk) -> None:
        """평상시 청크를 워커에 넘긴다(즉시 반환). 큐가 차 있으면 버린다 — 실시간 우선."""
        try:
            self._q.put_nowait(("chunk", chunk))
        except queue.Full:
            pass

    def reset(self) -> None:
        """긴급 진입 → 모은 발화 버퍼를 비우라고 워커에 지시(즉시 반환)."""
        try:
            self._q.put_nowait(("reset", None))
        except queue.Full:
            pass

    def latest(self) -> Optional[SpeechResult]:
        """워커가 완성한 자막을 한 번 꺼낸다(꺼내면 비워짐). 없으면 None."""
        with self._lock:
            r, self._latest = self._latest, None
            return r

    def stop(self) -> None:
        self._stop.set()
        try:
            self._q.put_nowait(("stop", None))
        except queue.Full:
            pass

    # ------------------------------------------------------------------
    def _run(self) -> None:
        """워커 루프: 큐에서 명령을 받아 처리. transcribe 블로킹은 이 스레드 안에서만."""
        while not self._stop.is_set():
            try:
                kind, data = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            if kind == "stop":
                break
            if kind == "reset":
                self._t.reset()
                continue
            res = self._t.transcribe(data)        # ← 블로킹(메인 아닌 이 스레드에서)
            if res.is_speech and res.text:
                with self._lock:
                    self._latest = res
