"""HUD를 위한 고속 방향(DoA) 추정 폴러.

파이프라인은 분류 정확도를 위해 1초 단위로 동작하지만, HUD 시각 반응(특히 레이더)은
1초마다 갱신되면 끊겨 보인다. 메인 오디오 캡처 스트림에서 0.25초 단위의 슬라이스를
넘겨받아 백그라운드 스레드에서 방향만 빠르게 갱신한다.
"""

from __future__ import annotations

import queue
import threading
from typing import Optional

import numpy as np

from core.types import DirectionResult, AudioChunk, Direction
from doa.estimator import estimate_direction


class DoaPoller:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        # 가장 최신 오디오만 처리하면 되므로 큐 크기는 작게 유지
        self._q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=3)
        self._latest: Optional[DirectionResult] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="doa_poller", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        # 큐 대기를 깨기 위해 더미 데이터 푸시
        try:
            self._q.put_nowait(np.zeros((1, 6), dtype=np.float32))
        except queue.Full:
            pass
        self._thread.join(timeout=1.0)

    def push(self, data: np.ndarray) -> None:
        """캡처 스레드에서 0.25초 슬라이스를 밀어넣는다 (블로킹 X)."""
        if self._q.full():
            try:
                self._q.get_nowait()  # 큐가 꽉 차면 오래된 것 버림
            except queue.Empty:
                pass
        self._q.put(data)

    def latest(self) -> Optional[DirectionResult]:
        """HUD가 매 프레임(30fps) 호출해 최신 방향을 가져간다."""
        with self._lock:
            return self._latest

    def _run(self) -> None:
        from pipeline.runner import _raw4
        
        has_multi = False
        try:
            from doa.multi_source import estimate_multiple_directions
            has_multi = True
        except ImportError:
            pass

        while not self._stop.is_set():
            try:
                data = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
                
            if self._stop.is_set():
                break

            # pipeline/runner.py 의 _direction 과 동일한 로직
            raw4 = _raw4(data)
            res = None
            
            if raw4 is not None and has_multi:
                try:
                    from doa.multi_source import estimate_multiple_directions
                    results = estimate_multiple_directions(raw4, fs=self.sample_rate)
                    if results:
                        angle, direction = results[0]
                        res = DirectionResult(direction=direction, angle_deg=angle)
                    else:
                        res = DirectionResult(direction=Direction.UNKNOWN)
                except Exception:
                    pass
            
            if res is None:
                # 1채널이거나 모듈 에러 시 폴백
                mono = data[:, 0] if data.ndim == 2 else data
                chunk = AudioChunk(samples=mono, sample_rate=self.sample_rate)
                res = estimate_direction(chunk)

            with self._lock:
                self._latest = res
