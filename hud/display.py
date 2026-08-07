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

    def set_doa_poller(self, poller) -> None:
        """별도 고속 DoA 폴러 연결."""
        self._doa_poller = poller

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
        pygame.mouse.set_visible(False)
        # 전체화면에서 SDL 이 요청 모드를 못 주면 더 큰 표면을 돌려준다. 설정값이 아니라
        # 실제 표면 크기로 버퍼를 잡아야 반사(상하반전) 모드에서 화면 전체가 뒤집힌다.
        # 렌더러는 draw() 에서 표면 크기를 보고 스스로 레이아웃을 다시 잡는다.
        real = self._screen.get_size()
        if real != (self._config.width, self._config.height):
            print(f"[hud] 요청 {self._config.width}x{self._config.height} → 실제 "
                  f"{real[0]}x{real[1]} — 실제 크기로 레이아웃을 잡습니다.", file=sys.stderr)
        self._buf = pygame.Surface(real)
        self._renderer = Renderer(self._config)

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
        with self._lock:
            fused = self._latest
            
        if fused is not None and self._doa_poller is not None:
            latest_dir = self._doa_poller.latest()
            if latest_dir is not None:
                fused = replace(fused, direction=latest_dir)
                
        view = HudView.from_fused(fused) if fused is not None else None
        try:
            if self._config.reflect:
                self._renderer.draw(self._buf, view)
                self._screen.blit(
                    pygame.transform.flip(self._buf, False, True), (0, 0))
            else:
                self._renderer.draw(self._screen, view)
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
