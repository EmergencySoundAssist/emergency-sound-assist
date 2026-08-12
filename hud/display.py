"""HUD 창 생명주기 + 렌더 루프 (메인 스레드 전용).

update()는 다른 스레드(파이프라인)가 최신 FusedResult를 밀어넣는 유일한 창구(락).
run()은 메인 스레드에서 30fps로 최신값을 그리고 이벤트(종료/반사 토글)를 처리한다.
"""

from __future__ import annotations

import sys
import threading
from dataclasses import replace

import pygame

from hud.renderer import Renderer
from hud.viewmodel import HudView

# 수집 모드 키 → 라벨. 1/2/3/u 는 tools/tag_siren.py 의 배열과 동일(+키패드),
# n 은 오검출 확인(사이렌 아님 — hard negative).
_COLLECT_KEYS = {}
for _keys, _label in (
    (("K_1", "K_KP1"), "ambulance"),
    (("K_2", "K_KP2"), "police"),
    (("K_3", "K_KP3"), "fire"),
    (("K_u", "K_0", "K_KP0"), "unknown"),
    (("K_h", "K_4", "K_KP4"), "horn"),
    (("K_n",), "not_siren"),
):
    for _k in _keys:
        _COLLECT_KEYS[getattr(pygame, _k)] = _label


class HudDisplay:
    def __init__(self, config):
        self._config = config
        self._lock = threading.Lock()
        self._latest = None
        self._stop = threading.Event()
        self._screen = None
        self._buf = None
        self._renderer = None
        self._doa_poller = None
        self._collector = None

    def set_doa_poller(self, poller) -> None:
        """별도 고속 DoA 폴러 연결."""
        self._doa_poller = poller

    def set_collector(self, collector) -> None:
        """수집 모드(--collect): 차종 버튼 오버레이 + 키/터치 입력을 이 수집기로 보낸다."""
        self._collector = collector

    def update(self, fused) -> None:
        """파이프라인 스레드가 최신 결과를 밀어넣음(블로킹 X)."""
        with self._lock:
            self._latest = fused

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def stop(self) -> None:
        self._stop.set()

    def _init_pygame(self) -> None:
        pygame.init()
        flags = pygame.FULLSCREEN if self._config.fullscreen else 0
        self._screen = pygame.display.set_mode(
            (self._config.width, self._config.height), flags)
        pygame.display.set_caption("Emergency Sound Assist HUD")
        # 수집 모드에선 커서를 남긴다 — 개발 창에서 마우스로 버튼을 눌러야 한다
        # (터치스크린은 커서가 보여도 해가 없다).
        pygame.mouse.set_visible(self._collector is not None)
        # 전체화면에서 SDL 이 요청 모드를 못 주면 더 큰 표면을 돌려준다. 설정값이 아니라
        # 실제 표면 크기로 버퍼를 잡아야 반사(상하반전) 모드에서 화면 전체가 뒤집힌다.
        # 렌더러는 draw() 에서 표면 크기를 보고 스스로 레이아웃을 다시 잡는다.
        real = self._screen.get_size()
        if real != (self._config.width, self._config.height):
            print(f"[hud] 요청 {self._config.width}x{self._config.height} → 실제 "
                  f"{real[0]}x{real[1]} — 실제 크기로 레이아웃을 잡습니다.", file=sys.stderr)
        self._buf = pygame.Surface(real)
        self._renderer = Renderer(self._config)

    def _collect_input(self, event) -> None:
        """수집 모드 입력: 키(1/2/3/u·z) + 버튼 터치/클릭.

        반사(상하반전) 모드에선 화면에 뒤집혀 그려지므로, 터치 y 를 되돌려서
        렌더러가 기억하는 버튼 사각형(뒤집기 전 좌표)과 맞춘다.
        """
        if event.type == pygame.KEYDOWN:
            label = _COLLECT_KEYS.get(event.key)
            if label is not None:
                self._collector.on_label(label)
            elif event.key == pygame.K_z:
                self._collector.on_cancel()
            return
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # SDL 은 터치에서 합성 마우스 이벤트(touch=True)를 추가로 만든다.
            # 거르지 않으면 탭 1회가 FINGERDOWN+MOUSEBUTTONDOWN 으로 두 번
            # 라벨을 찍는다(직전 클립 라벨 + 유령 수동 녹음 개시).
            if getattr(event, "touch", False):
                return
            pos = event.pos
        elif event.type == pygame.FINGERDOWN:       # 터치스크린은 정규화 좌표로 온다
            w, h = self._screen.get_size()
            pos = (int(event.x * w), int(event.y * h))
        else:
            return
        if self._config.reflect:
            pos = (pos[0], self._screen.get_size()[1] - 1 - pos[1])
        for rect, key in self._renderer.collect_buttons:
            if rect.collidepoint(pos):
                self._collector.on_label(key)
                return

    def _tick(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.stop()
                return
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self.stop()
                    return
                if event.key == pygame.K_f:
                    self._config.reflect = not self._config.reflect
            if self._collector is not None:
                self._collect_input(event)
        with self._lock:
            fused = self._latest

        if fused is not None and self._doa_poller is not None:
            latest_dir = self._doa_poller.latest()
            if latest_dir is not None:
                fused = replace(fused, direction=latest_dir)

        view = HudView.from_fused(fused) if fused is not None else None
        collect = self._collector.status() if self._collector is not None else None
        try:
            if self._config.reflect:
                self._renderer.draw(self._buf, view, collect)
                self._screen.blit(
                    pygame.transform.flip(self._buf, False, True), (0, 0))
            else:
                self._renderer.draw(self._screen, view, collect)
        except Exception as e:          # 렌더 예외가 루프를 죽이지 않게 격리
            print(f"[hud] 렌더 예외(무시): {e}", file=sys.stderr)
        pygame.display.flip()

    def run(self) -> None:
        """메인 스레드 렌더 루프. stop() 또는 창 종료까지 블로킹."""
        try:
            self._init_pygame()
        except pygame.error as e:
            print(f"[hud] 디스플레이 초기화 실패: {e}\n"
                  "  HDMI 연결·그래픽 세션(DISPLAY)이 필요합니다.", file=sys.stderr)
            self.stop()
            return
        clock = pygame.time.Clock()
        try:
            while not self._stop.is_set():
                self._tick()
                clock.tick(self._config.fps)
        finally:
            pygame.quit()
