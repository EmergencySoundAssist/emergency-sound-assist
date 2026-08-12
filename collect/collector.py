"""통합 수집기 — 감지 파이프라인에 얹혀 사이렌 이벤트 클립을 모은다.

기존 tools/tag_siren.py 는 HUD 를 세우고(마이크 점유) 세션 전체를 녹음한 뒤
tools/cut_clips.py 로 잘랐다. 이 모듈은 반대로 **평소처럼 감지+HUD 를 돌리면서**:

  - 감지가 사이렌이라고 하는 순간 클립을 연다(자동). 링버퍼(프리롤) 덕에
    감지보다 먼저 울린 소리도 클립 머리에 들어간다.
  - 조수석이 차종 버튼(HUD 키 1/2/3/u · 화면 터치)을 누르면 라벨이 붙는다.
    라벨은 소리보다 늦게 온다(차를 보고 누르기까지 ~10초) — 그래서 버튼은
    **가장 오래 라벨을 기다리는 쪽부터** 채운다: 방금 닫힌 미라벨 클립(grace)
    → 녹음 중 클립. 사이렌은 들리는데 차를 못 봤으면 u(차종모름)를 눌러 순서를
    지킬 것. 아무 대상도 없으면 감지가 놓친 사이렌으로 보고 **수동 클립**을
    연다(미검출 표본 — 검출기 개선용).
  - 잘못 눌렀으면 z: 마지막 라벨을 취소한다(녹음 중 → 직전 클립 순).

분류 판정은 **디바운스 전(raw)** 을 받는다. 파이프라인의 잔향(hangover)은
벽시계 기준이라, 실시간보다 빨리 도는 소스(--wav 재수집)에서 오디오 시간축과
어긋난다. 깜빡임은 auto_tail(오디오 3초)이 자체적으로 메운다.

시간축은 전부 '흘러간 오디오 샘플 수'로 센다(tag_siren 과 동일). 벽시계는
세션 폴더명·t_wall 기록과 피드백 표시 시한에만 쓴다.

산출:
  <out>/<날짜시각(초)>_<장소>/clips/NNN_{auto|manual}.wav   (16k mono int16)
  <out>/<날짜시각(초)>_<장소>/labels.csv
라벨은 파일명이 아니라 labels.csv 에만 있다 — 나중에 누른 라벨(grace)이 파일
이름을 바꾸지 않게 하기 위해서다. csv 는 임시파일+원자 교체로 매번 다시 쓴다.
presses 열은 녹음 중 눌린 (클립 내 시각:라벨) 목록 — 연속 차량이 한 클립으로
병합됐을 때 tools/cut_collect.py 가 구간별로 라벨을 나눠 붙이는 근거다.
5초 학습 클립으로의 절단은 tools/cut_collect.py 가 맡는다.
"""
from __future__ import annotations

import csv
import os
import threading
import time
import wave
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from core.types import AudioChunk, ClassResult, SirenSubtype, SoundClass

LABEL_UNLABELED = "unlabeled"
LABEL_NOT_SIREN = "not_siren"     # 오검출 확인 — 검출은 울렸는데 사이렌도 경적도 아니었음
LABEL_HORN = "horn"               # 경적 — 사이렌과 함께 긴급으로 HUD 를 띄우는 클래스

# HUD 피드백 문구용 (수집기는 화면에 낼 문구까지 만든다 — HUD 는 그대로 그린다)
_KO = {
    SirenSubtype.AMBULANCE.value: "구급차",
    SirenSubtype.POLICE.value: "경찰차",
    SirenSubtype.FIRE.value: "소방차",
    SirenSubtype.UNKNOWN.value: "차종모름",
    LABEL_HORN: "경적",
    LABEL_NOT_SIREN: "사이렌아님",
}


def _session_dir(out_root: str | Path, place: str) -> Path:
    """초 단위 타임스탬프 + 충돌 시 접미사 — 재시작이 직전 세션을 절대 덮지 않게.

    분 단위면 같은 분 안의 재시작(크래시 후 곧장 다시 켜는 흔한 상황)이 기존
    폴더를 재사용해 labels.csv 를 비우고 wav 를 덮어쓴다.
    """
    base = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{place}"
    d = Path(out_root) / base
    n = 1
    while d.exists():
        n += 1
        d = Path(out_root) / f"{base}_{n}"
    return d


class SirenCollector:
    """감지 결과·버튼 입력을 받아 이벤트 클립 wav + labels.csv 를 쌓는다.

    스레드: on_result() 는 파이프라인 스레드, on_label()/on_cancel()/status() 는
    HUD 메인 스레드에서 온다. 내부 락 하나로 전부 지킨다. close() 는 멱등이라
    파이프라인 finally 와 메인 스레드 양쪽에서 불러도 안전하다(먼저 온 쪽만
    저장·요약을 수행).

    pre_roll    : 트리거 앞쪽으로 살려 둘 오디오(초) — 링버퍼 크기.
    auto_tail   : 자동 클립에서 사이렌 판정이 끊긴 뒤 더 담는 꼬리(초).
    manual_sec  : 수동 클립이 버튼 이후 담는 길이(초). 마감 때 사이렌 판정이
                  살아 있으면 자동 클립처럼 끝날 때까지 연장한다.
    max_sec     : 클립 길이 상한 — 판정이 눌어붙어도 파일이 폭주하지 않게.
    grace       : 클립이 닫힌 뒤에도 라벨 키가 그 클립을 가리키는 시간(초).
    clock       : 피드백 표시 시한용 단조 시계(테스트 주입용).
    """

    FEEDBACK_SECONDS = 4.0

    def __init__(
        self,
        out_root: str | Path = "data/collect_sessions",
        place: str = "미지정",
        pre_roll: float = 10.0,
        auto_tail: float = 3.0,
        manual_sec: float = 20.0,
        max_sec: float = 120.0,
        grace: float = 15.0,
        clock=time.monotonic,
    ) -> None:
        safe = place.replace("/", "-").replace(" ", "") or "미지정"
        self.session = _session_dir(out_root, safe)
        (self.session / "clips").mkdir(parents=True, exist_ok=True)
        self._place = safe
        self._pre_roll = float(pre_roll)
        self._auto_tail = float(auto_tail)
        self._manual_sec = float(manual_sec)
        self._max_sec = float(max_sec)
        self._grace = float(grace)
        self._clock = clock

        self._lock = threading.Lock()
        self._closed = False
        self._sr: Optional[int] = None        # 첫 청크에서 확정
        self._pos = 0.0                       # 지금까지 흘러간 오디오(초) — 내부 시간축
        self._ring: deque[np.ndarray] = deque()
        self._ring_sec = 0.0
        self._clip: Optional[dict] = None     # 열린 클립 (없으면 None)
        self._rows: list[dict] = []           # labels.csv 행 (통째 재작성용)
        self._last_closed_at: Optional[float] = None   # 직전 클립이 닫힌 _pos
        self._feedback: Optional[tuple[str, float]] = None   # (문구, clock 시각)
        self._write_csv()                     # 빈 파일이라도 먼저 — 경로 오타를 즉시 드러낸다

    # ── 파이프라인 스레드 ────────────────────────────────────────────────

    def on_result(self, chunk: AudioChunk, cls: ClassResult) -> None:
        """매 틱(1초 청크) 호출. **디바운스 전(raw)** 분류 결과를 받는다."""
        mono = _channel0(chunk)
        siren = bool(cls.is_emergency and cls.label is SoundClass.SIREN)
        # 경적도 긴급으로 HUD 를 띄우므로 같이 모은다. 안 모으면 경적 쪽 지연·오검출은
        # 데이터가 없어 분석 자체가 불가능하다(1차 수집에서 실제로 그랬다).
        horn = bool(cls.is_emergency and cls.label is SoundClass.HORN)
        with self._lock:
            if self._closed:
                return
            if self._sr is None:
                self._sr = int(chunk.sample_rate)
            dur = mono.size / float(self._sr)
            self._pos += dur

            if self._clip is None:
                self._ring_push(mono, dur)
                if siren or horn:
                    # 사이렌·경적 어느 쪽이든 클립을 연다. 무엇이 열었는지는
                    # trigger_class 로 남겨, 나중에 두 문제를 갈라서 볼 수 있게 한다.
                    self._open_clip(trigger="auto", label=LABEL_UNLABELED, cls=cls,
                                    trigger_class="siren" if siren else "horn")
                return

            c = self._clip
            c["frames"].append(mono)
            c["samples"] += mono.size
            c["flags"].append(siren)
            c["horn_flags"].append(horn)
            if siren:
                c["last_siren"] = self._clip_sec(c)
                if cls.confidence > c["conf_max"]:
                    c["conf_max"] = cls.confidence
                sc = cls.subtype_confidence or 0.0
                if cls.subtype is not None and sc >= c["model_sub_conf"]:
                    c["model_subtype"], c["model_sub_conf"] = cls.subtype.value, sc
            if horn:
                c["last_siren"] = self._clip_sec(c)     # 경적도 클립을 살려 둔다
                # ⚠ conf_max(=det_conf_max) 는 건드리지 않는다. 하류 도구
                # (tools/export_hardneg._screen, tools/cut_collect)가 이 값을
                # '사이렌다움'으로 읽으므로, 경적 확신도를 섞으면 경적 클립이
                # 사이렌급으로 보여 스크린 판단이 틀어진다.
                if cls.confidence > c["horn_conf_max"]:
                    c["horn_conf_max"] = cls.confidence

            sec = self._clip_sec(c)
            over = sec >= self._max_sec
            if c["trigger"] == "auto":
                done = sec - c["last_siren"] >= self._auto_tail
            else:   # manual: 목표 길이를 채웠고, 사이렌 판정이 살아 있지 않으면 닫는다
                done = sec - c["pre_roll"] >= self._manual_sec and not siren
            if done or over:
                self._close_clip()

    # ── HUD 스레드 ──────────────────────────────────────────────────────

    def on_label(self, label: "str | SirenSubtype") -> None:
        """라벨 버튼. 라벨을 가장 오래 기다린 쪽부터: 직전 미라벨 클립(grace) →
        녹음 중 클립 → (둘 다 없으면) 수동 녹음 시작.

        순서가 이래야 하는 이유: 라벨은 사이렌보다 ~10초 늦는다. 연속 출동에서
        앞차 클립이 닫히고 뒷차 클립이 이미 열린 뒤에 앞차 라벨이 도착하는데,
        열린 클립을 먼저 채우면 앞차 라벨이 뒷차에 붙는다(체계적 오라벨).
        버튼이 차량 순서대로 눌린다는 약속이므로, 차를 못 봤어도 u 를 눌러
        순서를 지켜야 한다. 잘못 붙인 라벨은 z 로 취소 후 다시 누른다.

        not_siren(오검출 확인)만 예외로 수동 녹음을 **열지 않는다** — '사이렌이
        아니다'는 소리는 새로 녹음할 대상이 없고, 이 버튼의 일은 오검출 클립에
        도장을 찍어 라벨 대기 줄에서 빼는 것뿐이다.
        """
        if isinstance(label, SirenSubtype):
            label = label.value
        ko = _KO.get(label)
        if ko is None:
            return
        with self._lock:
            if self._closed:
                self._say("세션 종료됨 — 입력은 기록되지 않는다")
                return
            row = self._graced_row()
            if row is not None and row["label"] == LABEL_UNLABELED:
                row["label"] = label
                self._write_csv()
                self._say(f"라벨 ✓ {ko} (직전 클립)")
                return
            if self._clip is not None:
                c = self._clip
                c["presses"].append((self._clip_sec(c), label))
                c["label"] = label
                self._say(f"라벨 ✓ {ko} (녹음 중 클립)")
                return
            if label in (LABEL_NOT_SIREN, LABEL_HORN):
                # 둘 다 '지금 들리는 사이렌을 새로 녹음한다'는 뜻이 아니다.
                # not_siren=오검출 도장, horn=경적 확인 — 대상 클립이 없으면 할 일이 없다.
                self._say(f"{ko}: 대상 클립 없음")
                return
            self._open_clip(trigger="manual", label=label, cls=None)
            self._say(f"수동 녹음 ● {ko} — 미검출 표본")

    def on_cancel(self) -> None:
        """z: 마지막 라벨 취소(녹음 중 클립 → 직전 클립 순). 파일은 지우지 않는다."""
        with self._lock:
            if self._closed:
                self._say("세션 종료됨 — 입력은 기록되지 않는다")
                return
            c = self._clip
            if c is not None and c["label"] != LABEL_UNLABELED:
                if c["presses"]:
                    c["presses"].pop()
                c["label"] = c["presses"][-1][1] if c["presses"] else LABEL_UNLABELED
                self._say("라벨 취소 (녹음 중 클립)")
                return
            row = self._graced_row()
            if row is not None and row["label"] != LABEL_UNLABELED:
                row["label"] = LABEL_UNLABELED
                self._write_csv()
                self._say("라벨 취소 (직전 클립)")
                return
            self._say("취소할 라벨 없음")

    def status(self) -> dict:
        """HUD 오버레이용 스냅샷."""
        with self._lock:
            counts: dict[str, int] = {}
            for r in self._rows:
                counts[r["label"]] = counts.get(r["label"], 0) + 1
            fb = None
            if self._feedback and self._clock() - self._feedback[1] < self.FEEDBACK_SECONDS:
                fb = self._feedback[0]
            c = self._clip
            return {
                "recording": c["trigger"] if c else None,
                "elapsed": self._clip_sec(c) if c else 0.0,
                "label": c["label"] if c else None,
                "feedback": fb,
                "counts": counts,
                "clips": len(self._rows),
                "session": self.session.name,
                "closed": self._closed,
            }

    def close(self) -> Optional[str]:
        """종료 — 열린 클립을 저장하고 요약 한 줄을 돌려준다.

        멱등: 처음 부른 쪽만 저장·요약을 얻고, 이후 호출은 None. 파이프라인
        스레드(데몬)가 종료 동결로 저장을 못 마칠 수 있어, main 이 join 후
        메인 스레드에서 한 번 더 부르는 것을 전제로 한다.
        """
        with self._lock:
            if self._closed:
                return None
            self._closed = True
            if self._clip is not None:
                self._close_clip()
            counts: dict[str, int] = {}
            for r in self._rows:
                counts[r["label"]] = counts.get(r["label"], 0) + 1
            parts = " · ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "클립 없음"
            return f"[collect] 클립 {len(self._rows)}개 ({parts}) → {self.session}"

    # ── 내부 (호출자가 락을 쥔 상태) ─────────────────────────────────────

    def _clip_sec(self, c: dict) -> float:
        return c["samples"] / float(self._sr or 1)

    def _ring_push(self, mono: np.ndarray, dur: float) -> None:
        """방금 청크를 넣고, '방금 청크 이전' 이력이 pre_roll 을 넘는 만큼만 버린다.

        pre_roll 은 트리거 **앞쪽** 오디오 약속이므로 방금 청크는 셈에서 뺀다 —
        포함해 버리면 자동 트리거의 실제 프리롤이 한 청크(1초) 짧아진다.
        """
        self._ring.append(mono)
        self._ring_sec += dur
        sr = float(self._sr or 1)
        while (self._ring
               and self._ring_sec - dur - self._ring[0].size / sr >= self._pre_roll):
            self._ring_sec -= self._ring.popleft().size / sr

    def _open_clip(self, trigger: str, label: str, cls: Optional[ClassResult],
                   trigger_class: str = "") -> None:
        """링버퍼 전체를 머리에 넣고 클립을 연다. 링은 비운다(클립과 중복 저장 방지).

        자동 트리거: 링 마지막 원소가 방금 판정난 청크다 → pre_roll 은 그 앞까지.
        직전 클립이 닫힌 직후라 링이 얕으면 프리롤도 그만큼 짧다(중복 없음이 우선).
        """
        frames = list(self._ring)
        samples = int(sum(f.size for f in frames))
        sr = float(self._sr or 1)
        cur = frames[-1].size / sr if (trigger == "auto" and frames) else 0.0
        sub = cls.subtype.value if (cls and cls.subtype is not None) else ""
        self._clip = {
            "trigger": trigger,
            "label": label,
            "frames": frames,
            "samples": samples,
            # det_flags 는 트리거 틱부터 기록한다 (프리롤 틱들의 판정은 링에 없다)
            "flags": [trigger_class == "siren"] if trigger == "auto" else [],
            "horn_flags": [trigger_class == "horn"] if trigger == "auto" else [],
            "trigger_class": trigger_class,
            "presses": [],                    # 녹음 중 눌린 (클립 내 초, 라벨)
            "pre_roll": samples / sr - cur,
            "last_siren": samples / sr,       # 트리거 시점 = 마지막 사이렌 관측
            "conf_max": (cls.confidence if (cls and trigger_class != "horn") else 0.0),
            "horn_conf_max": (cls.confidence if (cls and trigger_class == "horn") else 0.0),
            "model_subtype": sub,
            "model_sub_conf": (cls.subtype_confidence or 0.0) if cls else 0.0,
            "t_wall": datetime.now().isoformat(timespec="seconds"),
        }
        self._ring.clear()
        self._ring_sec = 0.0

    def _close_clip(self) -> None:
        c, self._clip = self._clip, None
        if c is None or self._sr is None or not c["frames"]:
            return
        name = f"{len(self._rows):03d}_{c['trigger']}.wav"
        _write_wav(self.session / "clips" / name,
                   np.concatenate(c["frames"]), self._sr)
        self._rows.append({
            "clip": f"clips/{name}",
            "label": c["label"],
            "trigger": c["trigger"],
            "t_wall": c["t_wall"],
            "pre_roll_sec": round(c["pre_roll"], 1),
            "duration_sec": round(c["samples"] / self._sr, 1),
            "det_flags": "".join("1" if f else "0" for f in c["flags"]),
            "horn_flags": "".join("1" if f else "0" for f in c["horn_flags"]),
            "trigger_class": c["trigger_class"],
            "det_conf_max": round(c["conf_max"], 3),
            "horn_conf_max": round(c["horn_conf_max"], 3),
            "model_subtype": c["model_subtype"],
            "model_sub_conf": round(c["model_sub_conf"], 3),
            "presses": "|".join(f"{t:.1f}:{lab}" for t, lab in c["presses"]),
        })
        self._last_closed_at = self._pos
        self._write_csv()
        self._say(f"클립 저장 #{len(self._rows) - 1}"
                  + ("" if c["label"] != LABEL_UNLABELED else " — 라벨 대기"))

    def _graced_row(self) -> Optional[dict]:
        """grace 안에 닫힌 직전 클립의 csv 행. 없으면 None."""
        if not self._rows or self._last_closed_at is None:
            return None
        if self._pos - self._last_closed_at > self._grace:
            return None
        return self._rows[-1]

    def _write_csv(self) -> None:
        """임시파일 + 원자 교체. 종료 동결·크래시가 반쯤 쓴 labels.csv 를 남기지 않게."""
        cols = ["clip", "label", "trigger", "trigger_class", "t_wall", "pre_roll_sec",
                "duration_sec", "det_flags", "horn_flags", "det_conf_max",
                "horn_conf_max", "model_subtype", "model_sub_conf", "presses"]
        path = self.session / "labels.csv"
        tmp = self.session / "labels.csv.tmp"
        with tmp.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(self._rows)
        os.replace(tmp, path)

    def _say(self, msg: str) -> None:
        self._feedback = (msg, self._clock())


def _channel0(chunk: AudioChunk) -> np.ndarray:
    """다채널이면 ch0(처리채널)만 — pipeline 과 같은 규약."""
    s = np.asarray(chunk.samples)
    mono = s[:, 0] if s.ndim == 2 else s
    return np.ascontiguousarray(mono, dtype=np.float32)


def _write_wav(path: Path, x: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())
