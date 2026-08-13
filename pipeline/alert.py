"""
alert.py — 검출 tick → 안정 알림 상태기계 + 출력 싱크 (순수 파이썬, TRT 무관)

설계: docs 워크플로 스펙. softmax(1.000 포화) 대신 **로짓 마진**으로 판정,
디바운스+히스테리시스+hangover+K/M투표+리마인더. 켜기 쉽게/끄기 느리게(청각장애 안전).
출력은 Sink 인터페이스로 분리(콘솔 지금 / 진동·화면 향후). 속도는 미사용.

⚠ 임계값은 닮은꼴(음악/사이렌FX) hard-negative 평가 전까지 **placeholder**.
"""
from __future__ import annotations

import shutil
import sys
import time
import unicodedata
from collections import Counter, deque
from dataclasses import dataclass

from core.types import Motion

CLASSES = ("siren", "horn", "noise")
LABEL_KO = {"siren": "사이렌", "horn": "경적", "noise": ""}

# 마진(z[cls]-max(나머지)) 기준 기본 설정 — tick dt=0.5s 가정. ⚠ S0(닮은꼴) 캘리 전 placeholder.
CFG = {
    # τ_on 1.5 근거(2026-07-03 실주행 녹음): 저SNR 실환경(차내, 앞 구급차)서 마진이 0~2에
    # 살아 τ2.0은 25s+ 무경보(첫 경보 34s), τ1.5는 6.2s·커버리지 9→21s. FA 검증: in-domain
    # 비-siren max -3.7, 합성 닮은꼴(알람/음악/처프/노이즈) max -0.01 — 전부 미발화.
    "siren": dict(tau_on=1.5, tau_off=0.5, N_on=2, T_hang=2.5, K_vote=3, M_win=5, T_remind=3.0),
    "horn":  dict(tau_on=2.5, tau_off=1.0, N_on=2, T_hang=1.0, K_vote=2, M_win=3, T_remind=4.0),
    # 예비(PRE) 게이트 — 짧은 창(2s) 검출용: 빨리 켜지고(≈2.7s) 빨리 접음(hangover 1s),
    # 리마인더 없음(확정 채널이 담당). 5s 확정 게이트의 recall은 건드리지 않는다.
    "siren_fast": dict(tau_on=2.0, tau_off=0.5, N_on=2, T_hang=1.0, K_vote=3, M_win=5, T_remind=9999.0),
}
TAU_CRIT = 4.0   # 이 마진 이상 + 지속이면 CRITICAL (속도 무관)
CAPTION_HOLD_SECONDS = 5.0  # 대시보드에서 완성 자막을 읽을 수 있도록 유지하는 시간


class SubtypeVote:
    """차종 시간 다수결 — 경보 활성 동안 tick별 argmax를 투표(신뢰 미달 tick은 기권),
    다수 라벨만 표시해 tick간 라벨 튐(경찰↔구급 진동)을 억제. clear 시 reset."""

    def __init__(self, labels=("구급차", "경찰차", "소방차"), win: int = 24):
        self.labels = labels
        self.votes = deque(maxlen=win)               # (클래스 idx, 확신도) 저장(기권 미저장)
        self.n_seen = 0

    def add(self, probs, conf: float) -> None:
        self.n_seen += 1
        i = int(probs.argmax())
        if float(probs[i]) >= conf:
            self.votes.append((i, float(probs[i])))

    def winner(self) -> tuple[int | None, int, int]:
        """(다수 클래스 idx, 그 표수, 전체 유효표) — 유효표가 없으면 (None, 0, 0).

        문자열 label() 과 **같은 집계**를 쓴다. HUD 는 enum 이 필요하고 콘솔은 문자열이
        필요한데, 집계를 두 벌 두면 화면과 로그가 서로 다른 차종을 말하게 된다.
        """
        if not self.votes:
            return None, 0, 0
        i, c = Counter(v for v, _ in self.votes).most_common(1)[0]
        return i, c, len(self.votes)

    def winner_conf(self) -> float | None:
        """이긴 클래스에 던져진 표들의 **모델 확신도 평균**.

        표 비율(c/n)을 확신도 자리에 넣으면 안 된다 — 같은 창이 여러 번 세어졌을 때
        0.62 짜리 관측이 1.00 으로 보고된다. 여기서 돌려주는 건 모델이 실제로 말한 값이다.
        """
        i, _, _ = self.winner()
        if i is None:
            return None
        cs = [c for v, c in self.votes if v == i]
        return sum(cs) / len(cs)

    def label(self) -> str:
        if not self.votes:                           # 유효 투표 0 → 세분화 보류
            return f"긴급차량({self.n_seen}tick)"
        i, c, n = self.winner()
        return f"{self.labels[i]}({c}/{n}표)"

    def reset(self) -> None:
        self.votes.clear()
        self.n_seen = 0


class Gate:
    """1클래스 디바운스+히스테리시스+hangover+투표+리마인더 상태기계.
    update(margin) → dict(active, onset, remind, clear)."""

    def __init__(self, cfg: dict, dt: float):
        self.cfg, self.dt = cfg, dt
        self.state = "OFF"
        self.run = 0
        self.hang = 0.0
        self.since_remind = 0.0
        self.votes = deque(maxlen=cfg["M_win"])

    def update(self, margin: float) -> dict:
        c = self.cfg
        on_hi = margin >= c["tau_on"]
        on_lo = margin >= c["tau_off"]
        self.votes.append(1 if on_hi else 0)
        voted = sum(self.votes) >= c["K_vote"]
        prev = self.state

        if self.state in ("OFF", "RISING"):
            self.run = self.run + 1 if on_hi else 0
            if self.run >= c["N_on"] and voted:
                self.state = "ON"; self.hang = c["T_hang"]; self.since_remind = 0.0
            else:
                self.state = "RISING" if self.run > 0 else "OFF"
        else:                                            # ON / FALLING
            if on_lo:
                self.state = "ON"; self.hang = c["T_hang"]
            else:
                self.hang -= self.dt
                self.state = "FALLING" if self.hang > 0 else "OFF"
                if self.state == "OFF":
                    self.run = 0

        active = self.state in ("ON", "FALLING")
        onset = active and prev in ("OFF", "RISING")     # 비활성→활성 = 새 경보 (RISING 경유 포함)
        remind = False
        if active:
            self.since_remind += self.dt
            if self.since_remind >= c["T_remind"]:
                remind = True; self.since_remind = 0.0
        clear = (not active) and prev in ("ON", "FALLING")
        return dict(active=active, onset=onset, remind=remind, clear=clear)


@dataclass(frozen=True)
class AlertEvent:
    level: str        # NONE / WARN / CRITICAL
    kind: str         # siren / horn / none
    label: str        # 한글
    margin: float     # 로짓 마진 (softmax 아님)
    onset: bool       # 새 경보 시작
    remind: bool      # ON 지속 리마인더 펄스
    clear: bool       # 해제
    subtype: str | None = None   # "긴급차량(0.71)" tier=ood, 기본 None
    risk: str | None = None      # 위험도 tier(정지/접근-느림/접근-빠름) — 속도엔진 있을 때만


def build_event(kind: str, margin: float, gate: dict | None, subtype: str | None = None,
                risk: str | None = None, tau_crit: float = TAU_CRIT,
                pre: bool = False) -> AlertEvent:
    """게이트 판정 → 표시 이벤트. **레벨은 margin 기반**(확률 임계 폐기).
    pre=True면 짧은 창 예비경보 — 레벨 PRE 고정(확정 전 단계, 진동 짧게)."""
    if kind == "siren":
        if pre:
            level = "PRE"
        else:
            level = "CRITICAL" if margin >= tau_crit else "WARN"   # 속도 무관, 마진+지속
        return AlertEvent(level, "siren", LABEL_KO["siren"], margin,
                          gate["onset"], gate["remind"], gate["clear"], subtype, risk)
    if kind == "horn":
        return AlertEvent("WARN", "horn", LABEL_KO["horn"], margin,
                          gate["onset"], gate["remind"], gate["clear"], None, None)
    return AlertEvent("NONE", "none", "", 0.0, False, False,
                      gate["clear"] if gate else False, None, None)


# ── 출력 싱크 ──────────────────────────────────────────────────────────────
def _proximity_bar(gauge: float | None, dir_raw: int | None, n: int = 10) -> str:
    """연속 근접 게이지(0~1) → '근접[████░░░░]↑' 막대. 다가오면 차오르고 멀어지면 빠짐.
    화살표: dir_raw 1=접근(↑)/2=멀어짐(↓). 꽉 차면(최근접) MAX⚠. gauge None이면 빈 문자열."""
    if gauge is None:
        return ""
    k = max(0, min(n, round(gauge * n)))
    bar = "█" * k + "░" * (n - k)
    arrow = {1: "↑", 2: "↓"}.get(dir_raw, " ")
    mx = " MAX⚠" if k >= n else ""
    return f"  근접[{bar}]{arrow}{mx}"


def _dwidth(s: str) -> int:
    """표시열 폭 — CJK(한글 등)는 2열로 센다."""
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in s)


def _fit_cols(s: str) -> str:
    """터미널 폭에 맞게 표시열 기준으로 자른다 — 줄바꿈·깨짐 방지(좁은 화면 대비)."""
    maxc = shutil.get_terminal_size((100, 24)).columns - 1
    if _dwidth(s) <= maxc:
        return s
    out, w = [], 0
    for c in s:
        cw = 2 if unicodedata.east_asian_width(c) in ("W", "F") else 1
        if w + cw > maxc:
            break
        out.append(c)
        w += cw
    return "".join(out)


def _status_line(margin: float, state: str, level: str, risk: str | None = None,
                 dir_raw: int | None = None, gauge: float | None = None) -> str:
    """연속 실시간 상태줄(제자리 갱신용). raw 마진 막대 + 게이트 상태 + 근접 게이지 + 위험도.
    막대: 마진 -2(빈칸)~+10(꽉) 12칸, τ_on≈4칸 지점. raw라 매 tick 흔들림(=실시간).
    gauge = 음량 기반 연속 근접도(0~1) — 다가오면 차오르고 멀어지면 빠지는 막대.
    좁은 터미널에서 줄이 넘치면 _fit_cols 가 뒤(위험도 텍스트)부터 자른다 — 근접 게이지는 앞쪽이라 보존.
    ('지금=' 은 게이지 화살표(↑/↓)·위험도 방향과 중복이라 제거)."""
    n = max(0, min(8, round((margin + 2.0) / 12.0 * 8)))    # 검출 막대 8칸(좁은 화면 대비 축소)
    bar = "▓" * n + "░" * (8 - n)
    st = {"OFF": "대기 ", "RISING": "↑감지", "ON": "●경보", "FALLING": "↓유지"}.get(state, state)
    active = state in ("ON", "FALLING")
    lv = f" {level}" if active else ""
    g = _proximity_bar(gauge, dir_raw) if active else ""   # 근접 게이지 — 우선 표시(앞쪽, 안 잘림)
    r = f"  위험도={risk}" if (active and risk) else ""    # 좁으면 이 뒤부터 잘림
    return _fit_cols(f"사이렌 [{bar}] {margin:+5.1f}  {st}{lv}{g}{r}")


class ConsoleSink:
    """연속 상태줄(매 tick 제자리 갱신=실시간) + 경보 엣지(ONSET/CLEAR 영구 줄). tty면 색."""
    _COLOR = {"CRITICAL": "\033[1;31m", "WARN": "\033[1;33m", "PRE": "\033[1;33m", "NONE": "\033[2m"}

    def __init__(self):
        self.tty = sys.stdout.isatty()
        self.t0 = time.time()

    def emit(self, e: AlertEvent) -> None:
        if not (e.onset or e.remind or e.clear):
            return
        tag = "ONSET" if e.onset else ("REMIND" if e.remind else "CLEAR")
        risk = f"  위험도={e.risk}" if e.risk else ""
        sub = f"  차종={e.subtype}(잠정)" if e.subtype else ""
        msg = f"[{time.time()-self.t0:6.1f}] {tag:6s} {e.level:8s} {e.label or '해제':4s}  margin={e.margin:+.2f}{risk}{sub}"
        if self.tty:
            msg = self._COLOR.get(e.level, "") + msg + "\033[0m"
        # 엣지=영구 줄. tty면 진행 중 상태줄을 지우고(\r\033[K) 새 줄에 출력 → 다음 tick이 그 아래에 상태줄.
        print(("\r\033[K" if self.tty else "") + msg, flush=True)

    def tick(self, margin: float, state: str, level: str, risk: str | None = None,
             dir_raw: int | None = None, gauge: float | None = None) -> None:
        """매 tick 연속 상태줄을 제자리 갱신 — 실시간 피드백. tty 아니면 생략(파이프 도배 방지)."""
        if not self.tty:
            return
        active = state in ("ON", "FALLING")
        col = self._COLOR.get(level if active else "NONE", "")
        sys.stdout.write("\r\033[K" + col
                         + _status_line(margin, state, level, risk, dir_raw, gauge) + "\033[0m")
        sys.stdout.flush()

    def close(self):
        if self.tty:
            sys.stdout.write("\r\033[K"); sys.stdout.flush()   # 잔여 상태줄 정리


_DIR_KO_FULL = {"front": "전방", "rear": "후방", "left": "좌측", "right": "우측", "unknown": "미상"}
_MOTION_KO = {Motion.APPROACHING: "접근 중", Motion.RECEDING: "멀어짐",
              Motion.STEADY: "유지", Motion.UNKNOWN: "판단중"}


class DashboardSink:
    """제자리 갱신 대시보드 — 스크롤 대신 한 판이 실시간으로 바뀐다(보기 쉬움).
    run_stream 이 매 tick update(ev, info)만 부른다. tty 아니면 조용(파이프 도배 방지).

    긴급:  종류·차종·레벨 / 방향 / 움직임 / 근접 게이지
    평상시: 마지막 자막을 5초 유지  ·  워밍업: 대기 중"""

    _COL = {"CRITICAL": "\033[1;31m", "WARN": "\033[1;33m", "PRE": "\033[1;33m"}
    _TOP = "━━━━━ 🚨 긴급음 감지 ━━━━━━━━━━━━━━━━━━"
    _BOT = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    def __init__(self, caption_hold_seconds: float = CAPTION_HOLD_SECONDS):
        self.tty = sys.stdout.isatty()
        self._lines = 0
        self._caption_hold_seconds = max(0.0, float(caption_hold_seconds))
        self._caption_text = ""
        self._caption_until = 0.0

    def update(self, ev, info) -> None:
        lines = self._render(ev, info)
        if not self.tty:
            return
        buf = f"\033[{self._lines}A" if self._lines else ""
        buf += "\n".join("\033[K" + ln for ln in lines) + "\n\033[J"
        sys.stdout.write(buf)
        sys.stdout.flush()
        self._lines = len(lines)

    def close(self) -> None:
        if self.tty:
            sys.stdout.write("\033[J")
            sys.stdout.flush()

    def _render(self, ev, info) -> list:
        rst = "\033[0m" if self.tty else ""
        if ev.kind != "none":                       # 긴급 화면에 이전 평상시 자막을 남기지 않음
            self._clear_caption()
        if not info:                                   # 워밍업(버퍼 부족)
            return [self._TOP, "  ⏳ 대기 중…", self._BOT]
        if ev.kind == "none":                          # 평상시 — STT 자막
            sp = info.get("speech")
            stt_status = info.get("stt_status") or {}
            if stt_status.get("last_error"):
                self._clear_caption()
                cap = f"  ⚠ STT 오류  {stt_status['last_error']}"
            else:
                now = time.monotonic()
                if sp is not None and getattr(sp, "is_speech", False) and sp.text:
                    self._caption_text = sp.text
                    self._caption_until = now + self._caption_hold_seconds
                if self._caption_text and now < self._caption_until:
                    cap = f'  💬 자막  "{self._caption_text}"'
                else:
                    self._clear_caption()
                    cap = "  💬 자막  (음성 없음)"
            dropped = int(stt_status.get("dropped_chunks", 0))
            queue_line = f"  ⚠ STT 지연  누적 {dropped}개 입력 생략" if dropped else None
            return [self._TOP, "  🟢 평상시", cap, *([queue_line] if queue_line else []), self._BOT]

        # 긴급 (사이렌/경적)
        icon = {"siren": "🚨 사이렌", "horn": "📢 경적"}.get(ev.kind, ev.label)
        col = self._COL.get(ev.level, "") if self.tty else ""
        sub = f"  {ev.subtype.split('(')[0]}" if ev.subtype else ""
        head = f"  {col}{icon}{rst}{sub}   {col}[{ev.level}]{rst}"

        d = info.get("direction")
        if d is not None:
            ang = f" ({info['angle']:.0f}°)" if info.get("angle") is not None else ""
            dline = f"  방향   {_DIR_KO_FULL.get(d.value, d.value)}{ang}"
        else:
            dline = "  방향   미상 (ReSpeaker 6채널 필요)"

        m = info.get("motion")
        mline = f"  움직임  {_MOTION_KO.get(m, '판단중')}"

        g = info.get("gauge")
        if g is not None:
            k = max(0, min(10, round(g * 10)))
            prox = info.get("proximity") or ""
            mark = " ⚠" if prox == "최근접" else ""
            gline = f"  근접   [{'█' * k}{'░' * (10 - k)}] {prox}{mark}"
        else:
            gline = "  근접   [░░░░░░░░░░]"

        return [self._TOP, head, dline, mline, gline, self._BOT]

    def _clear_caption(self) -> None:
        self._caption_text = ""
        self._caption_until = 0.0


class GpioSink:
    """진동(1차)+LED. Jetson.GPIO lazy. ⚠ 미완성 스텁 — 핀 활성화(DTO)·PWM 검증 필요."""
    def __init__(self, vib_pin: int = 33):
        import Jetson.GPIO as GPIO   # noqa: lazy, Jetson 전용
        self.GPIO, self.vib = GPIO, vib_pin
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(vib_pin, GPIO.OUT, initial=GPIO.LOW)
        self.selftest()

    def selftest(self):
        self.GPIO.output(self.vib, self.GPIO.HIGH); time.sleep(0.3)
        self.GPIO.output(self.vib, self.GPIO.LOW)

    def emit(self, e: AlertEvent) -> None:
        if e.onset or e.remind:
            dur = 0.6 if e.level == "CRITICAL" else 0.3
            self.GPIO.output(self.vib, self.GPIO.HIGH); time.sleep(dur)
            self.GPIO.output(self.vib, self.GPIO.LOW)

    def close(self):
        self.GPIO.cleanup()


class MultiSink:
    def __init__(self, sinks):
        self.sinks = sinks

    def emit(self, e):
        for s in self.sinks:
            try:
                s.emit(e)
            except Exception as ex:                       # 하나 죽어도 나머지 계속
                print(f"[sink 오류] {type(s).__name__}: {ex}", file=sys.stderr)

    def tick(self, *a, **k):                              # 연속 상태줄 — tick 지원 싱크에만 전달
        for s in self.sinks:
            f = getattr(s, "tick", None)
            if f is None:
                continue
            try:
                f(*a, **k)
            except Exception:
                pass

    def close(self):
        for s in self.sinks:
            try:
                s.close()
            except Exception:
                pass


def make_sink(spec: str) -> MultiSink:
    """'console' / 'console,gpio' 등 콤마 문자열 → MultiSink."""
    reg = {"console": ConsoleSink, "gpio": GpioSink}
    sinks = []
    for name in (s.strip() for s in spec.split(",") if s.strip()):
        if name not in reg:
            raise ValueError(f"알 수 없는 sink: {name} (가능: {list(reg)})")
        sinks.append(reg[name]())
    return MultiSink(sinks or [ConsoleSink()])


# ── Gate 단위 테스트 (TRT 없이 합성 마진으로) ─────────────────────────────
if __name__ == "__main__":
    dt = 0.5
    g = Gate(CFG["siren"], dt)
    # 마진 시퀀스: 잡음(낮음) → 사이렌(높음) 지속 → 멎음. onset/remind/clear 확인
    seq = [0.1, 0.2, 3.0, 3.5, 3.2, 4.5, 4.0, 3.8, 3.1, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0]
    print(f"{'t':>4s} {'margin':>7s} {'state':>8s}  events")
    for i, m in enumerate(seq):
        r = g.update(m)
        ev = [k for k in ("onset", "remind", "clear") if r[k]]
        print(f"{i*dt:4.1f} {m:7.2f} {g.state:>8s}  active={int(r['active'])} {ev}")
