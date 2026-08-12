"""마진 게이트 — 격자를 촘촘하게 해도 오탐이 폭증하지 않게 하는 장치.

## 왜 필요한가

검출 격자를 1.0초 → 0.25초로 줄이면 지연은 줄지만(중앙 4.00 → 3.12초) **초당 발화
기회도 4배**가 되어 오탐 클립이 14/61 → 22/61 로 늘어난다(tools/bench_runtime.py 실측).
틱 단위 오탐률은 5.4% 로 일정한데, 기회가 늘어 클립당 한 번은 걸리게 되는 것이다.
현장에서 "계속 사이렌"으로 나타났다.

Airacle deploy 런타임이 stride 0.15초로 돌 수 있는 이유가 여기 있다 — 격자가 촘촘해서
견디는 게 아니라 **게이트가 고립된 튐을 걸러내기** 때문이다(deploy alert.py CFG).
격자만 옮기고 게이트를 안 옮기면 오탐만 늘어난다.

## 규칙 (deploy alert.Gate 이식, 시간상수는 전부 '초')

  켜기: 마진이 tau_on 이상인 틱이 **연속 t_on 초** 이상 && 최근 t_vote 초 창에서
        k_frac 비율 이상 → ON
  끄기: 마진이 tau_off(< tau_on) 아래로 떨어져도 hangover 동안은 유지 → FALLING → OFF

켜기는 어렵게(높은 임계·연속·투표), 끄기는 느리게(낮은 임계·hangover). 비대칭이 핵심이다.
경보를 늦추지 않으면서 튐만 걸러내려는 구조다.

시간상수를 **초로 두고 격자로 환산**하므로, 격자를 바꿔도 실효 동작이 유지된다.
격자가 굵으면(1.0초) 투표창이 1틱으로 붕괴해 게이트가 사실상 무력해진다 — 그래서
게이트는 촘촘한 격자와 짝으로만 의미가 있다.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class GateConfig:
    tau_on: float          # 켜기 임계 (로짓 마진)
    tau_off: float         # 유지 임계 (히스테리시스, tau_on 보다 낮다)
    t_on: float            # 연속으로 tau_on 을 넘겨야 하는 시간(초)
    t_vote: float          # 투표창 길이(초)
    k_frac: float          # 투표창에서 필요한 상회 비율
    t_hang: float          # tau_off 아래로 떨어진 뒤 유지하는 시간(초)


# deploy alert.py CFG 와 같은 값 — 실차에서 튜닝된 것이라 출발점으로 그대로 쓴다.
SIREN = GateConfig(tau_on=1.2, tau_off=0.5, t_on=0.3, t_vote=0.75, k_frac=0.6, t_hang=2.5)
HORN = GateConfig(tau_on=2.5, tau_off=1.0, t_on=0.3, t_vote=0.45, k_frac=0.6, t_hang=1.0)


class Gate:
    """마진 시계열 → 켜짐/꺼짐. step(margin) 을 매 틱 부르고 bool 을 받는다."""

    def __init__(self, cfg: GateConfig, dt: float) -> None:
        self.cfg, self.dt = cfg, float(dt)
        self.n_on = max(1, round(cfg.t_on / self.dt))
        self.m_win = max(1, round(cfg.t_vote / self.dt))
        self.k_vote = min(self.m_win, max(self.n_on, round(cfg.k_frac * self.m_win)))
        self._run = 0
        self._votes: deque[int] = deque(maxlen=self.m_win)
        self._hang = 0.0
        self.on = False

    def step(self, margin: float) -> bool:
        hi = margin >= self.cfg.tau_on
        self._votes.append(1 if hi else 0)
        self._run = self._run + 1 if hi else 0
        if not self.on:
            if self._run >= self.n_on and sum(self._votes) >= self.k_vote:
                self.on, self._hang = True, self.cfg.t_hang
        else:
            if margin >= self.cfg.tau_off:
                self._hang = self.cfg.t_hang        # 재충전
            else:
                self._hang -= self.dt
                if self._hang <= 0.0:
                    self.on, self._run = False, 0
        return self.on

    def reset(self) -> None:
        self._run = 0
        self._votes.clear()
        self._hang = 0.0
        self.on = False


class EmergencyGate:
    """사이렌·경적 두 채널을 묶어 '긴급인가' 하나로 답한다."""

    def __init__(self, dt: float, siren: GateConfig = SIREN, horn: GateConfig = HORN) -> None:
        self.siren = Gate(siren, dt)
        self.horn = Gate(horn, dt)

    def step(self, m_siren: float, m_horn: float) -> bool:
        s = self.siren.step(m_siren)
        h = self.horn.step(m_horn)
        return s or h

    def reset(self) -> None:
        self.siren.reset()
        self.horn.reset()
