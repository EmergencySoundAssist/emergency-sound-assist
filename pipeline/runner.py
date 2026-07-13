"""
하이브리드 실시간 파이프라인 — 석우(ViT deploy) 경보 엔진 + 우리 방향/STT.  (exp/hybrid-runtime)

  chunk(0.15s tick) → 검출 마진(5s 확정 + 2s 예비 87f) → alert 상태기계(Gate)
    긴급 활성:  차종 다수결(SubtypeVote, yt판) + 접근/멀어짐(음량 기울기 approach.detector)
               + 방향(SRP-PHAT, ch1~4 롤링 0.5s)  ·  STT reset(발화 버퍼 비움)
    평상시   :  STT — 0.15s 청크를 ★1초 뭉치로 모아 워커에 feed
               (silence_release 가 청크 개수 기준이라, 잘게 넣으면 숨쉬기에 발화가 잘림 — 확정 결정)

  process(chunk) → (alert.AlertEvent, info)
    info: m_siren/state/level/risk/dir_raw(즉시)/direction/angle/speech/label/conf

lean(feat/lean-integration) 대비 변경:
  매 tick FusedResult → 상태기계 이벤트(ONSET/REMIND/CLEAR, 켜기 쉽게/끄기 느리게)
  접근/멀어짐 = 음량 기울기(approach.detector) — 신경망 dirhead 제거(원리 투명·모델 불필요).
    음량 커지는 추세=접근·작아지는 추세=멀어짐(소스 음량 상쇄 → 스피커 크기 불변).
    clf.speed_dir()/alert.SpeedTracker 경로는 미사용(코드 보존 — 되돌리기 쉽게).
  차종 부활 (yt 실채널 파인튜닝판 + 6초 다수결)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from core.types import AudioChunk, Motion, SAMPLE_RATE
from classifier import inference as clf
from approach.detector import ApproachDetector
from pipeline import alert

SUBS = ("구급차", "경찰차", "소방차")     # subtype_clf.SUBS 순서 (yt 모델 동일)
DIR_WIN_S = 0.5                           # 방향(SRP) 분석 창 — 다채널 롤링 버퍼 길이(초)

# 접근/멀어짐(음량 기울기) → 상태줄 '지금=' 즉시 표시용 인덱스 (alert.DIR_KO 순서).
# 유지/미상은 즉시 화살표를 띄우지 않는다(정지 vs 유지 혼동 회피).
_MOTION_DIR_IDX = {Motion.APPROACHING: 1, Motion.RECEDING: 2}


def _approach_tier(motion, level, proximity=None):
    """approach.Motion(+빠르기 1~5, +상대 근접도) → 위험도 tier 문자열. 미상/무관이면 None.
    접근일 때 음량 기울기 크기로 빠르기 1~5단계를 그대로 표시(예: '접근(빠르기4)').
    근접도(최근접/근거리/원거리)는 최근접 이후에만 붙는다 (예: '멀어짐·근거리')."""
    if motion is Motion.APPROACHING:
        base = f"접근(빠르기{level})" if level else "접근"
    elif motion is Motion.RECEDING:
        base = "멀어짐"
    elif motion is Motion.STEADY:
        base = "유지"
    else:
        return None
    return f"{base}·{proximity}" if proximity else base


class Pipeline:
    """청크(0.15s 권장) 단위 처리기. process(chunk) → (AlertEvent, info).

    stt_worker: stt.worker.STTWorker (선택). None 이면 자막 없음.
    dt: tick 간격(초) — Gate/SubtypeVote 시간상수를 dt 무관하게 유지.
    """

    def __init__(self, stt_worker=None, dt: float = 0.15) -> None:
        self.dt = dt
        self._stt = stt_worker
        self.g_siren = alert.Gate(alert.CFG["siren"], dt)
        self.g_horn = alert.Gate(alert.CFG["horn"], dt)
        self.g_fast = alert.Gate(alert.CFG["siren_fast"], dt)
        # 접근/멀어짐 = 음량 기울기(approach.detector) — 신경망 dirhead 대체(원리 투명·모델 불필요).
        # 자체 3초 추세 윈도우로 스무딩하므로 SpeedTracker(tick 스무딩)는 불필요.
        self.approach = ApproachDetector(sample_rate=SAMPLE_RATE)
        # 시간상수(차종투표≈6s) 유지 — 석우 UnifiedRuntime 과 동일 계산
        self.svote = alert.SubtypeVote(SUBS, win=max(8, round(6.0 / dt)))
        self._stt_acc: list = []                  # ★ STT 1초 뭉치 누적
        self._stt_len = 0
        self._dir_buf: Optional[np.ndarray] = None   # (n,4) 방향용 롤링
        self._dir_sr: Optional[int] = None

    def process(self, chunk: AudioChunk):
        s = np.asarray(chunk.samples)
        mono = s[:, 0] if s.ndim == 2 else s      # ch0 = 처리채널 (검출·차종·속도·STT)
        self._push_dir(s, chunk.sample_rate)      # ch1~4 = 방향용 롤링

        res = clf.analyze(AudioChunk(samples=mono, sample_rate=chunk.sample_rate))
        if res is None:                           # 워밍업(버퍼 부족)
            return alert.build_event("none", 0.0, None), {}

        sg = self.g_siren.update(res["m_siren"])
        hg = self.g_horn.update(res["m_horn"])
        fg = self.g_fast.update(res["m_fast"]) if res["m_fast"] is not None else None

        sub = risk = None
        dir_idx = None
        gauge = None
        ap_motion = ap_speed = ap_prox = None         # 대시보드용 구조화 값
        pre = False
        if sg["active"]:                          # 우선순위: 확정 siren > 예비(PRE) > horn
            kind, margin, gate = "siren", res["m_siren"], sg
            # 음량 기울기 → 접근/멀어짐/유지 (+빠르기·근접도·연속 게이지). 사이렌 활성 동안만 판정.
            ap = self.approach.update(AudioChunk(samples=mono, sample_rate=chunk.sample_rate))
            risk = _approach_tier(ap.motion, ap.speed_level, ap.proximity)  # +상대 근접도
            dir_idx = _MOTION_DIR_IDX.get(ap.motion)    # 상태줄 '지금=' 즉시 표시용(접근/멀어짐만)
            gauge = ap.gauge                            # 연속 근접 게이지(0~1) — 막대 표시용
            ap_motion, ap_speed, ap_prox = ap.motion, ap.speed_level, ap.proximity
            if self.g_siren.state == "ON":              # FALLING(꺼진 꼬리) 중엔 새 투표 없음
                sp = clf.subtype_probs()
                if sp is not None:
                    self.svote.add(sp, clf.SUBTYPE_CONF)
            if self.svote.n_seen:
                sub = self.svote.label()                # 다수결 라벨 (단일 tick 아님)
        elif fg is not None and fg["active"]:
            kind, margin, gate, pre = "siren", res["m_fast"], fg, True   # 예비경보(짧은 창)
        elif hg["active"]:
            kind, margin, gate = "horn", res["m_horn"], hg
        else:
            kind, margin = "none", 0.0
            gate = {"onset": False, "remind": False,
                    "clear": sg["clear"] or hg["clear"] or bool(fg and fg["clear"])}
        if sg["clear"]:                            # 경보 해제 → 다음 경보 위해 리셋
            self.approach.reset()
            self.svote.reset()

        emergency = sg["active"] or hg["active"] or bool(fg and fg["active"])
        direction, angle = self._direction() if emergency else (None, None)
        speech = self._stt_step(mono, chunk.sample_rate, emergency)

        ev = alert.build_event(kind, margin, gate, sub, risk, pre=pre)
        info = dict(m_siren=res["m_siren"], state=self.g_siren.state, level=ev.level,
                    risk=risk, dir_raw=dir_idx, gauge=gauge, direction=direction, angle=angle,
                    motion=ap_motion, speed_level=ap_speed, proximity=ap_prox,
                    speech=speech, label=res["label"], conf=res["conf"])
        return ev, info

    # ------------------------------------------------------------------
    def _push_dir(self, s: np.ndarray, sr: int) -> None:
        """다채널이면 ch1~4(원본 마이크)를 방향용 롤링 버퍼(0.5s)에 누적."""
        if s.ndim != 2:
            return
        if s.shape[1] >= 5:
            raw4 = s[:, 1:5]
        elif s.shape[1] == 4:
            raw4 = s
        else:
            return
        if self._dir_buf is None or self._dir_sr != sr:
            self._dir_buf = np.zeros((0, 4), dtype=np.float32)
            self._dir_sr = sr
        n = int(DIR_WIN_S * sr)
        self._dir_buf = np.concatenate([self._dir_buf, raw4.astype(np.float32)])[-n:]

    def _direction(self):
        """롤링 4채널 → SRP-PHAT 방향. 채널/의존성 없으면 (None, None) — graceful."""
        if (self._dir_buf is None or self._dir_sr is None
                or len(self._dir_buf) < int(DIR_WIN_S * self._dir_sr)):
            return None, None
        try:
            from doa.multi_source import estimate_multiple_directions
            results = estimate_multiple_directions(self._dir_buf, fs=self._dir_sr)
        except Exception:
            return None, None
        if not results:
            return None, None
        angle, direction = results[0]              # 에너지 가장 센 방향
        return direction, angle

    def _stt_step(self, mono: np.ndarray, sr: int, emergency: bool):
        """긴급이면 reset, 평상시엔 ★1초 뭉치로 모아 feed. 완성 자막을 돌려준다."""
        if self._stt is None:
            return None
        if emergency:                              # 긴급 우선 — 모으던 발화도 버림
            self._stt.reset()
            self._stt_acc, self._stt_len = [], 0
            return None
        self._stt_acc.append(np.asarray(mono, dtype=np.float32))
        self._stt_len += len(mono)
        if self._stt_len >= sr:                    # 1초 모임 → 워커로 (tick 크기와 무관)
            bundle = np.concatenate(self._stt_acc)
            self._stt_acc, self._stt_len = [], 0
            self._stt.feed(AudioChunk(samples=bundle, sample_rate=sr))
        return self._stt.latest()
