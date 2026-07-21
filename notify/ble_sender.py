"""
Jetson = BLE 클라이언트(중앙기기). 워치(GATT 서버)를 찾아 **연결을 계속 유지**하며
protocol.encode() 가 만든 4바이트를 write 한다.

설계 (요구사항 ③ 지연 최소화):
  - 연결은 시작 시 1회 → 끊기면 자동 재연결. 매 전송마다 스캔/연결하지 않음.
  - send() 는 '최신 1건'만 유지(밀린 옛 데이터 버림) → 항상 가장 최근 상황 전송.
  - write_gatt_char(response=False) = Write Without Response → 응답 대기 없이 즉시.

main.py 는 동기 루프라, 여기선 백그라운드 스레드에서 asyncio 이벤트루프를 돌리고
send() 는 그 루프로 페이로드를 넘기는 '동기 → 비동기' 다리 역할만 한다.

사용:
  sender = BleSender()              # 서비스 UUID로 워치 자동 검색
  sender = BleSender(address="AA:BB:..")  # 또는 워치 MAC 직접 지정(더 빠름/확실)
  sender.start()
  ...  sender.send(fused)  ...
  sender.close()
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Optional

from core.types import FusedResult
from notify.protocol import encode, encode_alert, SERVICE_UUID, ALERT_CHAR_UUID, DEVICE_NAME

log = logging.getLogger("notify")


class BleSender:
    def __init__(
        self,
        address: Optional[str] = None,
        scan_timeout: float = 8.0,
        reconnect_delay: float = 2.0,
    ) -> None:
        self._address = address
        self._scan_timeout = scan_timeout
        self._reconnect_delay = reconnect_delay

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._queue: Optional[asyncio.Queue] = None
        self._stop = False
        self._last: Optional[bytes] = None      # 직전 전송 페이로드(중복 전송 방지)
        self.connected = False

    # ── 외부 API (동기) ───────────────────────────────────────────
    def start(self) -> None:
        """백그라운드 BLE 스레드 시작."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()

    def send(self, fused: FusedResult) -> None:
        """결과를 워치로 전송(비차단). 직전과 동일하면 보내지 않음."""
        self._enqueue(encode(fused))

    def send_alert(self, ev, info: dict) -> None:
        """approach-loudness 파이프라인용 전송(비차단). (AlertEvent, info) → 4바이트, 직전과 같으면 스킵.

        이 브랜치 main 은 FusedResult 대신 (ev, info) 를 다루므로 send() 대신 이걸 쓴다.
        소리 종류는 디바운스된 경보 상태(ev)를 따른다 — protocol.encode_alert 참고.
        """
        self._enqueue(encode_alert(ev, info))

    def _enqueue(self, payload: bytes) -> None:
        """4바이트를 BLE 스레드로 넘긴다(비차단). 직전과 같으면 스킵.

        스레드/루프가 아직 없거나 이미 닫혔으면 조용히 무시한다 —
        BLE 실패(예: bleak 미설치·어댑터 문제)가 감지 파이프라인을 멈추지 않게 한다.
        """
        if payload == self._last:               # 같은 상황 반복 → 스킵(진동/트래픽 폭주 방지)
            return
        loop = self._loop
        if loop is None or self._queue is None or loop.is_closed():
            return
        self._last = payload
        try:
            asyncio.run_coroutine_threadsafe(self._put_latest(payload), loop)
        except RuntimeError:                    # 루프가 막 닫힘(BLE 스레드 종료 등) → 무시
            pass

    def close(self) -> None:
        """종료. (스레드는 daemon 이라 프로세스 종료도 막지 않음)"""
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=self._scan_timeout + 1.0)

    # ── 내부 (비동기) ─────────────────────────────────────────────
    def _thread_main(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._queue = asyncio.Queue(maxsize=1)
        try:
            self._loop.run_until_complete(self._run())
        finally:
            self._loop.close()

    async def _put_latest(self, payload: bytes) -> None:
        # 큐는 항상 '최신 1개'만 유지: 밀린 데이터를 비우고 최신만 넣는다.
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        await self._queue.put(payload)

    async def _find_device(self):
        # 지연 import: bleak 미설치 환경(예: --ble 안 쓸 때)에서도 모듈 import 가능하게.
        from bleak import BleakScanner

        if self._address:
            return await BleakScanner.find_device_by_address(
                self._address, timeout=self._scan_timeout
            )

        def _match(_device, adv) -> bool:
            uuids = [u.lower() for u in (adv.service_uuids or [])]
            return SERVICE_UUID.lower() in uuids or adv.local_name == DEVICE_NAME

        return await BleakScanner.find_device_by_filter(_match, timeout=self._scan_timeout)

    async def _run(self) -> None:
        from bleak import BleakClient

        while not self._stop:
            try:
                log.info("워치 스캔 중…")
                device = await self._find_device()
                if device is None:
                    log.warning("워치를 찾지 못함 — 재시도")
                    await asyncio.sleep(self._reconnect_delay)
                    continue

                log.info("연결 시도: %s", device.address)
                async with BleakClient(device) as client:
                    self.connected = True
                    log.info("연결됨 — 전송 대기")
                    while not self._stop and client.is_connected:
                        try:
                            # 1초마다 깨어나 stop/연결상태 확인
                            payload = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                        except asyncio.TimeoutError:
                            continue
                        await client.write_gatt_char(ALERT_CHAR_UUID, payload, response=False)
            except Exception as e:  # 연결 끊김·스캔 실패 등 → 재연결 루프로
                log.warning("BLE 오류: %s", e)
            finally:
                self.connected = False

            if not self._stop:
                await asyncio.sleep(self._reconnect_delay)
