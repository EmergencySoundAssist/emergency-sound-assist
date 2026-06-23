"""② 방향 추정 — 2단계: 다중 음원 (동시 여러 방향).

raw 4채널 → SRP-PHAT 공간 스펙트럼 → 다중 peak 추출 → 방향들.

1단계([estimator.angle_to_direction])는 ReSpeaker 자체 DoA로 **단일 방향**만 구한다.
본 모듈은 raw 4채널을 직접 처리해 **동시 2개+ 방향**을 추정한다 (WBS 7주차 고도화).

설계 메모:
- `pick_peaks` 는 순수 numpy/scipy → 하드웨어·pyroomacoustics 없이 단위 테스트 가능.
- `spatial_spectrum` 만 pyroomacoustics 에 의존하며, 미설치 시 호출 시점에 명확히 안내한다.
- 4-mic 소형 어레이의 현실적 한계: 분리 가능한 동시 음원은 ~2개, 각도가 충분히
  떨어져 있어야 한다. 그 이상은 신뢰도가 급락한다.

⚠️ pyroomacoustics 방위각(반시계, 0°=+x)과 ReSpeaker raw DOA 규약이 다를 수 있으므로,
   `angle_to_direction` 에 넣기 전 **단일 음원으로 먼저 보정**할 것
   (한 방향에서만 소리 → peak 1개가 맞는 방향에 뜨는지 확인). docs/doa/direction-mapping.md 참고.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from core.types import Direction
from doa.estimator import angle_to_direction

# ---------------------------------------------------------------------------
# 마이크 어레이 기하 (보드: MIC1~4 = 원 위 45/135/225/315°)
# ---------------------------------------------------------------------------
# TODO(확인): ReSpeaker 4-mic 어레이 반지름(m)을 데이터시트/PCB 실측으로 확정할 것.
MIC_RADIUS_M = 0.046
MIC_ANGLES_DEG = (45.0, 135.0, 225.0, 315.0)  # MIC1, MIC2, MIC3, MIC4

FS_DEFAULT = 16000
NFFT_DEFAULT = 512
SIREN_BAND_HZ = (500.0, 1500.0)
SPEED_OF_SOUND = 343.0


def mic_locations(
    radius_m: float = MIC_RADIUS_M,
    angles_deg: Tuple[float, ...] = MIC_ANGLES_DEG,
) -> np.ndarray:
    """마이크 좌표 (2, M). SRP/MUSIC 에 넘길 평면 배치."""
    ang = np.deg2rad(np.asarray(angles_deg, dtype=float))
    return np.vstack([radius_m * np.cos(ang), radius_m * np.sin(ang)])


def pick_peaks(
    spectrum: np.ndarray,
    azimuth_deg: np.ndarray,
    max_src: int = 2,
    height_ratio: float = 0.5,
    min_sep_deg: float = 30.0,
) -> List[float]:
    """공간 스펙트럼에서 봉우리(방향)를 여러 개 추출한다. (순수 numpy/scipy)

    Args:
        spectrum:     각도별 에너지. azimuth_deg 와 같은 길이.
        azimuth_deg:  각 빈의 방위각(0~360, 오름차순 가정).
        max_src:      최대 음원 수 (에너지 큰 순 상위 N).
        height_ratio: 최대치 대비 임계 (이 비율 미만 봉우리는 무시).
        min_sep_deg:  봉우리 간 최소 각도 간격 (붙은 봉우리 병합 방지).

    Returns:
        에너지 내림차순으로 정렬된 방위각 리스트(도). 없으면 빈 리스트.
    """
    from scipy.signal import find_peaks

    s = np.asarray(spectrum, dtype=float)
    if s.size == 0 or not np.any(s > 0):
        return []
    s = s / s.max()

    n = s.size
    step = 360.0 / n                       # 빈당 각도 (그리드 균일 가정)
    distance = max(1, int(round(min_sep_deg / step)))

    # 원형(0°=360°) 경계 처리: 앞쪽 distance 개를 뒤에 덧붙여 wrap-around 봉우리도 포착
    wrapped = np.concatenate([s, s[:distance]])
    idx, _ = find_peaks(wrapped, height=height_ratio, distance=distance)
    idx = np.unique(idx % n)
    if idx.size == 0:
        return []

    # 에너지 큰 순으로 상위 max_src 개
    idx = sorted(idx, key=lambda i: s[i], reverse=True)[:max_src]
    return [float(azimuth_deg[i]) for i in idx]


def spatial_spectrum(
    raw4: np.ndarray,
    fs: int = FS_DEFAULT,
    nfft: int = NFFT_DEFAULT,
    freq_range: Tuple[float, float] = SIREN_BAND_HZ,
    algo: str = "SRP",
    num_src: int = 2,
    radius_m: float = MIC_RADIUS_M,
) -> Tuple[np.ndarray, np.ndarray]:
    """raw 4채널 → (방위각 그리드[도], 공간 스펙트럼). pyroomacoustics 의존.

    Returns:
        azimuth_deg: 0~360 오름차순 그리드 (길이 360)
        spectrum:    그리드별 에너지 (azimuth_deg 와 같은 길이)
    """
    try:
        import pyroomacoustics as pra
    except ImportError as e:  # pragma: no cover - 환경 의존
        raise ImportError(
            "pyroomacoustics 가 필요합니다: pip install pyroomacoustics"
        ) from e

    raw4 = np.asarray(raw4, dtype=float)
    if raw4.ndim != 2 or raw4.shape[1] != 4:
        raise ValueError(f"raw4 는 (n_samples, 4) 여야 함. 받은 shape={raw4.shape}")

    # 채널별 STFT → (4, n_freq, n_frames)
    X = np.array([
        pra.transform.stft.analysis(raw4[:, ch], nfft, nfft // 2).T
        for ch in range(4)
    ])

    grid = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    doa = pra.doa.algorithms[algo](
        mic_locations(radius_m), fs, nfft,
        c=SPEED_OF_SOUND, num_src=num_src, azimuth=grid,
    )
    doa.locate_sources(X, freq_range=list(freq_range))

    az_deg = np.degrees(doa.grid.azimuth) % 360.0
    spec = np.asarray(doa.grid.values, dtype=float)

    order = np.argsort(az_deg)             # %360 후 오름차순 정렬
    return az_deg[order], spec[order]


def estimate_multiple_directions(
    raw4: np.ndarray,
    fs: int = FS_DEFAULT,
    num_src: int = 2,
    freq_range: Tuple[float, float] = SIREN_BAND_HZ,
    nfft: int = NFFT_DEFAULT,
    height_ratio: float = 0.5,
    min_sep_deg: float = 30.0,
    radius_m: float = MIC_RADIUS_M,
) -> List[Tuple[float, Direction]]:
    """raw 4채널 → [(방위각, Direction), ...] 동시 여러 방향.

    각도는 SRP-PHAT 추정 raw 방위각이며, `angle_to_direction` 으로 4방향 매핑한다.
    (raw 각도 ↔ 차량 기준 보정은 estimator 의 보정 상수를 따른다 — 모듈 상단 ⚠️ 참고.)
    """
    az, spec = spatial_spectrum(
        raw4, fs=fs, nfft=nfft, freq_range=freq_range,
        num_src=num_src, radius_m=radius_m,
    )
    angles = pick_peaks(
        spec, az, max_src=num_src,
        height_ratio=height_ratio, min_sep_deg=min_sep_deg,
    )
    return [(a, angle_to_direction(a)) for a in angles]


def capture_raw4(seconds: float = 1.0, fs: int = FS_DEFAULT) -> np.ndarray:
    """ReSpeaker 6채널 중 ch1~4(원본)만 캡처. sounddevice 의존."""
    try:
        import sounddevice as sd
    except ImportError as e:  # pragma: no cover - 환경 의존
        raise ImportError("sounddevice 가 필요합니다: pip install sounddevice") from e
    data = sd.rec(int(seconds * fs), samplerate=fs, channels=6, dtype="float32")
    sd.wait()
    return np.asarray(data)[:, 1:5]


if __name__ == "__main__":
    # 실물 캡처 → 동시 다방향 추정 (ReSpeaker + pyroomacoustics 필요)
    print("raw 4채널 1초 캡처 → 다중 방향 추정")
    chunk = capture_raw4(1.0)
    for ang, direction in estimate_multiple_directions(chunk):
        print(f"  {ang:5.0f}° → {direction.value}")
