"""HUD 시나리오 데모 — 마이크·젯슨 없이 화면 동작을 눈으로 확인한다.

여러 상황을 이어서 반복 재생한다: 대기 → 멀리서 접근 → 좌우로 지나가며 방향 회전
→ 최근접 → 멀어짐 → 대기. 차종과 진입 방향을 바꿔가며 계속 돈다.

정지 렌더(tools/render_hud_states.py)로는 안 보이는 것을 본다:
  - 음압이 오르내릴 때 링이 차고 빠지는가
  - 차가 지나갈 때 아크가 실제로 회전하는가
  - 접근과 멀어짐이 화면에서 구분되는가

사용:
  python tools/hud_demo.py              # 창 모드
  python tools/hud_demo.py --fullscreen
  종료: ESC 또는 Q
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.types import (
    ApproachResult, ClassResult, Direction, DirectionResult, FusedResult,
    Motion, SirenSubtype, SoundClass, SpeechResult,
)
from doa.estimator import _to_vehicle_angle, angle_to_direction
from hud.config import HudConfig
from hud.display import HudDisplay

# 차량 기준 각도 → raw 방위각(장착 보정의 역함수). 1°  해상도면 화면상 충분하다.
_RAW_BY_VEHICLE = {
    v: min(range(360), key=lambda r: abs((_to_vehicle_angle(r) - v + 180) % 360 - 180))
    for v in range(0, 360, 2)
}


def _raw_for(vehicle_deg: float) -> float:
    return float(_RAW_BY_VEHICLE[int(round(vehicle_deg / 2) * 2) % 360])


def _fused(subtype, vehicle_deg, spl, motion, speech=None):
    raw = _raw_for(vehicle_deg)
    return FusedResult(
        sound=ClassResult.from_label(SoundClass.SIREN, 0.94, subtype),
        direction=DirectionResult(direction=angle_to_direction(raw), angle_deg=raw),
        approach=ApproachResult(motion=motion, level_db=spl),
        speech=speech,
    )


def _idle(speech=None):
    return FusedResult(
        sound=ClassResult.from_label(SoundClass.NORMAL_TRAFFIC, 0.8),
        direction=DirectionResult(direction=Direction.UNKNOWN),
        approach=ApproachResult(motion=Motion.UNKNOWN),
        speech=speech,
    )


# (차종, 진입 차량각, 통과 방향) — 진입각에서 시작해 ±180° 를 돌아 빠져나간다.
PASSES = [
    (SirenSubtype.AMBULANCE, 60.0, +1),    # 우전방 → 우 → 우후방
    (SirenSubtype.POLICE, 300.0, -1),      # 좌전방 → 좌 → 좌후방
    (SirenSubtype.FIRE, 20.0, +1),         # 거의 정면에서 시작
    (SirenSubtype.UNKNOWN, 200.0, -1),     # 후방에서 좌측으로
]
SPL_FAR, SPL_NEAR = 62.0, 104.0


def timeline():
    """(지속시간초, FusedResult 생성 함수) 를 무한히 내놓는다."""
    while True:
        for subtype, entry, sweep in PASSES:
            yield 2.2, lambda: _idle()
            yield 1.6, lambda: _idle(SpeechResult(text="사이렌이 들리기 시작합니다",
                                                  is_speech=True))
            # 접근: 각도는 천천히 돌고 음압은 오른다
            for i in range(26):
                u = i / 25
                deg = entry + sweep * 70.0 * u
                spl = SPL_FAR + (SPL_NEAR - SPL_FAR) * (u ** 1.6)
                yield 0.13, (lambda st=subtype, d=deg, s=spl:
                             _fused(st, d, s, Motion.APPROACHING))
            # 통과 직후: 각도가 빠르게 돌고 음압은 정점에서 꺾인다
            for i in range(14):
                u = i / 13
                deg = entry + sweep * (70.0 + 70.0 * u)
                spl = SPL_NEAR - 10.0 * u
                yield 0.10, (lambda st=subtype, d=deg, s=spl:
                             _fused(st, d, s, Motion.RECEDING))
            # 멀어짐
            for i in range(22):
                u = i / 21
                deg = entry + sweep * (140.0 + 40.0 * u)
                spl = (SPL_NEAR - 10.0) - (SPL_NEAR - 10.0 - SPL_FAR) * (u ** 0.8)
                yield 0.14, (lambda st=subtype, d=deg, s=spl:
                             _fused(st, d, s, Motion.RECEDING))
        yield 2.0, lambda: _idle(SpeechResult(text="잠시만요, 길 좀 비켜 주시겠어요?",
                                              is_speech=True))


def main() -> None:
    ap = argparse.ArgumentParser(description="HUD 시나리오 데모(마이크 불필요)")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--flip", action="store_true", help="반사(윈드실드) 모드로 시작")
    args = ap.parse_args()

    hud = HudDisplay(HudConfig(fullscreen=args.fullscreen, reflect=args.flip))

    import threading

    def feed():
        for hold, make in timeline():
            if hud.stopped:
                return
            hud.update(make())
            end = time.monotonic() + hold
            while time.monotonic() < end:
                if hud.stopped:
                    return
                time.sleep(0.02)

    threading.Thread(target=feed, name="demo-feed", daemon=True).start()
    print("== HUD 데모: 접근 → 통과 → 멀어짐 을 차종·방향 바꿔가며 반복 ==")
    print("   종료: ESC 또는 Q")
    hud.run()


if __name__ == "__main__":
    main()
