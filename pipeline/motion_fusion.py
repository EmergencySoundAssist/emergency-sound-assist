"""속도 방향 모델·음량 추세·직접 도플러를 조건부로 선택하는 C 융합.

고정 가중 평균을 하지 않는다. 서로 다른 증거는 값의 의미와 오차 분포가 달라서
평균할 수 없기 때문이다. 공개 사이렌 10개 원본을 이용한 source-wise 교차검증으로
아래 문턱을 선택했으며, 직접 도플러는 사이렌 자체 변조에 취약하므로 충돌을 푸는
보조 근거로만 사용한다.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

from classifier.inference import SpeedEvidence
from core.types import ApproachResult, Motion


DIR_MOTIONS = (Motion.STEADY, Motion.APPROACHING, Motion.RECEDING)

# evaluation/tune_fusion.py의 leave-one-source-out 최빈 선택값(10개 중 8개 fold 일치).
MODEL_CONFIDENCE_MIN = 0.55
DOPPLER_CONFIDENCE_MIN = 0.20


@dataclass(frozen=True)
class FusionDecision:
    motion: Motion
    source: str


def conditional_decision(
    model: SpeedEvidence | None,
    acoustic: ApproachResult,
) -> FusionDecision:
    """한 시점의 조건부 C 판단을 반환한다.

    모델의 5초 창이 아직 준비되지 않았거나 모델 파일이 없을 때는 음량 판단으로
    안전하게 폴백한다. 모델이 있어도 신뢰도, 음량 변화 강도, 직접 도플러의 유효도에
    따라 증거를 선택하며 숫자 평균은 하지 않는다.
    """
    a = acoustic.motion
    acoustic_valid = a is not Motion.UNKNOWN
    if model is None:
        return FusionDecision(a, "acoustic_warmup" if acoustic_valid else "none")

    m = DIR_MOTIONS[model.direction_index]
    model_valid = model.confidence >= MODEL_CONFIDENCE_MIN
    doppler_confidence = acoustic.doppler_confidence or 0.0
    doppler_valid = doppler_confidence >= DOPPLER_CONFIDENCE_MIN
    doppler_motion = acoustic.doppler_motion

    if acoustic_valid and m is a:
        return FusionDecision(m, "agree")
    if model_valid and not acoustic_valid:
        return FusionDecision(m, "model_only")
    if not model_valid and not acoustic_valid:
        return FusionDecision(Motion.UNKNOWN, "none")

    # 충돌할 때는 모델과 독립 도플러가 동시에 같은 쪽을 지지해야만 음량 판단을 뒤집는다.
    # 직접 도플러 단독 성능이 낮았기 때문에 이보다 공격적인 override는 허용하지 않는다.
    if model_valid and doppler_valid and doppler_motion is m:
        return FusionDecision(m, "model+doppler")
    return FusionDecision(a, "acoustic_primary")


class ConditionalMotionFusion:
    """조건부 C 판단에 짧은 다수결을 적용해 tick 간 깜빡임을 억제한다."""

    def __init__(self, smooth_size: int = 3):
        self._buf: deque[FusionDecision] = deque(maxlen=max(1, int(smooth_size)))
        self._previous = Motion.UNKNOWN

    def reset(self) -> None:
        self._buf.clear()
        self._previous = Motion.UNKNOWN

    def update(self, model: SpeedEvidence | None, acoustic: ApproachResult) -> FusionDecision:
        raw = conditional_decision(model, acoustic)
        self._buf.append(raw)
        votes = Counter(d.motion for d in self._buf if d.motion is not Motion.UNKNOWN)
        if not votes:
            return FusionDecision(self._previous, "hold")
        motion, count = votes.most_common(1)[0]
        if count < 2 and self._previous is not Motion.UNKNOWN:
            motion = self._previous
        else:
            self._previous = motion
        return FusionDecision(motion, raw.source if motion is raw.motion else "smoothed")
