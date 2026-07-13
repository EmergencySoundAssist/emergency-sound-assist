"""
③ 접근/멀어짐 판단 모듈 — 실시간 도플러(주파수) + 음량(SPL) 추세.

설계: docs/approach/design.md
  - 음원이 접근/멀어지면 대표 주파수와 음량(에너지)이 시간에 따라 변한다.
  - 연속된 청크의 추세를 보고 motion(접근/멀어짐/유지)을 판정.

★ 물리 보정 (설계 문서 표의 '주파수↑=접근'은 직관적이지만 통과 상황에선 부정확):
  - 관측 주파수는 통과(pass-by) 내내 **단조 하강**(높→낮)한다. 따라서 주파수 '추세 부호'
    만으로는 접근/멀어짐을 가를 수 없다 (게다가 사이렌 원본 주파수를 모름).
  - 음량(SPL)은 가장 가까운 순간에 **최대**가 되므로, 접근=커짐 / 멀어짐=작아짐으로
    **부호가 깔끔히 뒤집힌다.** → 실시간 부호의 '주 신호'는 음량 추세.
  - 도플러(주파수)는 (a) 사이렌 톤 존재 게이트, (b) 강한 하강 글라이드=통과 진행 신호로
    음량을 '보강'한다. (docs/03 "진폭 추세는 부호 판별에 견고"와 일치)

핵심 설계:
  - **추세(기울기)의 부호**만 사용 → 사이렌 원본 주파수·절대 음압 레벨을 몰라도 동작.
    (log-에너지의 기울기는 소스 음압 레벨이라는 상수가 상쇄되어 '거리 변화율'만 남음.)
  - 매 청크 원본을 롤링 버퍼(window_seconds)에 쌓고, 겹치는 sub-frame들로
    (대표주파수, log-에너지) 시계열을 만들어 1차 선형회귀로 추세를 추정.
  - 관측이 충분히 쌓이기 전(버퍼 부족·톤 미검출)에는 UNKNOWN → 시간이 지나며 확정.

의존성: numpy, scipy (보드 친화적, 학습 모델 불필요).
"""

from __future__ import annotations

import numpy as np

from core.types import AudioChunk, ApproachResult, Motion, SAMPLE_RATE

# 사이렌 에너지 대역(Hz): 기본 톤 + 낮은 배음.
_BAND = (300.0, 2500.0)

# 접근 빠르기 단계 경계 (log-파워/초 → 1~5).  ← 튜닝 대상
# 음량 기울기의 '크기' = 다가오는 빠르기 (소스 음압 상쇄 → 스피커 크기 불변).
# 합성 실측: 60km/h·측면 8m 원거리 접근 구간 기울기 ~0.8 → 3단계.
SPEED_LEVEL_EDGES = (0.30, 0.60, 0.90, 1.30)

# 상대 근접도 경계 — 이벤트 내 '최고 음량(=최근접)' 대비 지금 위치.
# 물리: power ∝ 1/r² → log-파워 낙폭 Δ → 거리비 r/r_min = exp(Δ/2).
#   낙폭 CLOSE_DROP(≈3dB) 이하 = 최근접 부근, 거리비 FAR_RATIO 이상 = 원거리.
# ⚠ 절대 거리(m)가 아니라 '가장 가까웠던 순간 대비' 상대값. 경계는 튜닝 대상.
CLOSE_DROP = 0.7        # log-파워 낙폭(≈3dB) — 이 이하면 '최근접'
FAR_RATIO = 2.0         # 최근접 대비 거리비 — 이 이상이면 '원거리'
MIN_RISE = 1.5          # 이벤트 시작 대비 최고치가 이만큼(≈6.5dB) 커진 뒤에만 근접도 산출
                        # (먼 접근 초반 잡음 출렁임이 만드는 가짜 '최근접' 억제)
GAUGE_SPAN = 3.0        # 연속 게이지 만점 기준 log-파워 상승폭(≈13dB) — 시작 대비 이만큼 커지면 MAX
                        # (다가오면 차오르고 멀어지면 빠지는 막대. ⚠ 상대값 — 절대 거리 아님, 튜닝 대상)


def _speed_level(eslope: float) -> int:
    """음량 기울기(log-파워/초) → 접근 빠르기 1~5단계."""
    lvl = 1
    for edge in SPEED_LEVEL_EDGES:
        if eslope >= edge:
            lvl += 1
    return min(lvl, 5)


def _to_mono(samples: np.ndarray) -> np.ndarray:
    """(n,) 또는 (n, ch) → (n,) float64 모노."""
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim > 1:                       # 다채널이면 평균내서 모노로
        x = x.mean(axis=1)
    return x


def _parabolic(y_l: float, y_m: float, y_r: float) -> float:
    """3점 포물선 정점 보정량 δ∈(-0.5, 0.5) — sub-bin 주파수 정밀화."""
    denom = y_l - 2 * y_m + y_r
    return 0.5 * (y_l - y_r) / denom if denom != 0 else 0.0


def _dominant_freq_energy(frame: np.ndarray, sr: int, band, prominence: float):
    """한 sub-frame의 (대표 주파수[Hz], log-대역에너지) 반환.

    - 대표 주파수: 대역 내 최대 피크(포물선 보간). 뚜렷한 톤이 없으면 NaN.
    - log-에너지: 대역 파워의 로그 (음량 추세용 — 기울기에서 소스 레벨 상쇄).
    """
    n = frame.size
    if n < 8:
        return np.nan, -np.inf
    w = np.hanning(n)
    spec = np.abs(np.fft.rfft(frame * w))
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    m = (freqs >= band[0]) & (freqs <= band[1])
    if m.sum() < 3:
        return np.nan, -np.inf
    bspec = spec[m]
    bfreq = freqs[m]
    power = float(np.sum(bspec ** 2))
    loge = np.log(power + 1e-12)

    k = int(np.argmax(bspec))
    med = float(np.median(bspec)) + 1e-12
    if bspec[k] <= prominence * med:     # 뚜렷한 톤 없음 → 주파수 무효(에너지는 유효)
        return np.nan, loge
    if 0 < k < bspec.size - 1:
        d = _parabolic(bspec[k - 1], bspec[k], bspec[k + 1])
    else:
        d = 0.0
    df = bfreq[1] - bfreq[0]
    return float(bfreq[k] + d * df), loge


def _slope(t: np.ndarray, y: np.ndarray) -> float:
    """1차 선형회귀 기울기. 점이 부족하거나 시간 분산이 0이면 0."""
    if t.size < 2 or np.ptp(t) <= 0:
        return 0.0
    try:
        return float(np.polyfit(t, y, 1)[0])
    except (np.linalg.LinAlgError, ValueError):
        return 0.0


class ApproachDetector:
    """연속 청크를 받아 접근/멀어짐을 실시간 판단.

    update(chunk) 를 매 청크 호출. classifier가 siren일 때만 호출하면 효율적
    (비사이렌에서는 톤 게이트가 막아 UNKNOWN 반환).
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        window_seconds: float = 3.0,      # 추세를 보는 관측 윈도우
        frame_seconds: float = 0.5,       # sub-frame 길이
        hop_seconds: float = 0.25,        # sub-frame 간격
        band=_BAND,
        prominence: float = 2.5,          # 톤 검출 문턱(중앙값 대비 배수)
        energy_deadband: float = 0.10,    # log-파워/초 (이 안쪽은 '유지')  ← 튜닝 대상
        glide_threshold: float = 8.0,     # Hz/초, 강한 하강 글라이드 = 통과 진행  ← 튜닝 대상
        min_valid_frames: int = 4,        # 부호 확정에 필요한 최소 톤 프레임
        min_span_seconds: float = 1.5,    # 부호 확정에 필요한 최소 관측 시간
    ):
        self.sr = int(sample_rate)
        self.frame = int(frame_seconds * sample_rate)
        self.hop = int(hop_seconds * sample_rate)
        self.maxlen = int(window_seconds * sample_rate)
        self.band = band
        self.prominence = prominence
        self.energy_deadband = energy_deadband
        self.glide_threshold = glide_threshold
        self.min_valid_frames = min_valid_frames
        self.min_span_seconds = min_span_seconds
        self._buf = np.zeros(0, dtype=np.float64)
        self._last = Motion.UNKNOWN
        self._peak_loge = -np.inf          # 이벤트 내 최고 음량(=최근접 지점) 추적
        self._start_loge = None            # 이벤트 시작 음량 (의미있는 접근 판단 기준)

    def reset(self) -> None:
        """새 이벤트 시작 시 상태 초기화."""
        self._buf = np.zeros(0, dtype=np.float64)
        self._last = Motion.UNKNOWN
        self._peak_loge = -np.inf          # 근접도 기준(최고 음량)도 리셋
        self._start_loge = None

    def update(self, chunk: AudioChunk) -> ApproachResult:
        x = _to_mono(chunk.samples)
        self._buf = np.concatenate([self._buf, x])[-self.maxlen:]
        motion, level, prox, rel, gauge = self._decide()
        self._last = motion
        return ApproachResult(motion=motion, speed_level=level,
                              proximity=prox, rel_distance=rel, gauge=gauge)

    # ------------------------------------------------------------------
    def _decide(self):
        """→ (Motion, 빠르기 1~5|None, 근접도|None, 거리비|None).
        빠르기는 접근일 때만(기울기 크기), 근접도는 최근접 이후(peak 대비)."""
        if self._buf.size < self.frame:
            return Motion.UNKNOWN, None, None, None, None

        ts, fs, es = [], [], []
        for start in range(0, self._buf.size - self.frame + 1, self.hop):
            frame = self._buf[start:start + self.frame]
            f, e = _dominant_freq_energy(frame, self.sr, self.band, self.prominence)
            ts.append((start + self.frame / 2) / self.sr)
            fs.append(f)
            es.append(e)
        ts = np.asarray(ts)
        fs = np.asarray(fs)
        es = np.asarray(es)

        tone = ~np.isnan(fs)                      # 사이렌 톤이 잡힌 프레임
        if tone.sum() < self.min_valid_frames:
            return Motion.UNKNOWN, None, None, None, None  # 톤 관측 부족 (비사이렌/무음)
        if np.ptp(ts[tone]) < self.min_span_seconds:
            return Motion.UNKNOWN, None, None, None, None  # 관측 시간 부족 (아직 추세 불명)

        # 주 신호: log-에너지 추세(부호가 통과점에서 뒤집힘)
        e_ok = np.isfinite(es)
        eslope = _slope(ts[e_ok], es[e_ok])
        # 보강: 도플러 주파수 추세 (강한 하강 = 통과 진행)
        fslope = _slope(ts[tone], fs[tone])

        level = None
        if eslope > self.energy_deadband:
            motion = Motion.APPROACHING
            level = _speed_level(eslope)           # 기울기 크기 = 접근 빠르기 (1~5)
        elif eslope < -self.energy_deadband:
            motion = Motion.RECEDING
        else:
            motion = Motion.STEADY

        # 도플러 보강: 음량이 모호(STEADY)한데 강한 하강 글라이드면 통과 전이 구간.
        # (글라이드는 항상 하강이라 단독으로 접근/이탈을 가르지 못하므로 음량을 주신호로 둠.)
        if motion is Motion.STEADY and fslope < -self.glide_threshold:
            motion = Motion.STEADY  # 전이 구간 — 명시적으로 유지 (부호 단정 회피)

        cur_loge = float(es[e_ok][-1]) if e_ok.any() else None
        prox, rel, gauge = self._proximity(motion, cur_loge)
        return motion, level, prox, rel, gauge

    def _proximity(self, motion, cur_loge):
        """이벤트 내 최고 음량(=최근접) 대비 현재 → (근접도 라벨, 거리비, 연속 게이지).
        물리: power∝1/r² → log-파워 낙폭 Δ → r/r_min = exp(Δ/2). 접근 전/미상이면 라벨 None.
        게이지(0~1)는 시작 대비 현재 음량 — 접근 중에도 계속 차오른다.
        ⚠ 절대 거리(m)가 아님 — '가장 가까웠던 순간 대비' 상대값."""
        if cur_loge is None or motion is Motion.UNKNOWN:
            return None, None, None
        if self._start_loge is None:                # 이벤트 첫 유효 음량 = 접근 기준선
            self._start_loge = cur_loge
        rising = cur_loge > self._peak_loge
        if rising:                                  # 아직 최고치 갱신 중 = 계속 다가오는 중
            self._peak_loge = cur_loge
        # 연속 게이지: 시작 대비 음량 상승 → 다가오면 차오르고 멀어지면 빠짐 (접근 중에도 유효)
        gauge = float(np.clip((cur_loge - self._start_loge) / GAUGE_SPAN, 0.0, 1.0))
        if rising:
            return None, None, gauge                # 다가오는 중 — 라벨 보류(peak 미확정), 게이지만
        if self._peak_loge - self._start_loge < MIN_RISE:
            return None, None, gauge                # 아직 의미있는 접근 없음 → 가짜 최근접 억제
        # 음량이 더는 안 오름 = 최고치(최근접)를 지났다 (motion 라벨의 3초 지연과 무관하게 즉시 포착)
        drop = self._peak_loge - cur_loge           # ≥ 0 (지금이 최고면 0)
        if drop < CLOSE_DROP:
            return "최근접", 1.0, gauge
        ratio = float(np.exp(drop / 2.0))           # power∝1/r² → 최근접 대비 거리비
        return ("근거리" if ratio < FAR_RATIO else "원거리"), ratio, gauge


if __name__ == "__main__":  # 빠른 스모크 (무음 → UNKNOWN)
    det = ApproachDetector()
    silent = AudioChunk(samples=np.zeros(SAMPLE_RATE, dtype=np.float32))
    for _ in range(4):
        print(det.update(silent).motion)
