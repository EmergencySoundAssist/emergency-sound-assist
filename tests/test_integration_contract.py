"""doa.estimator ↔ 파이프라인 통합 계약 테스트.

기존 test_estimator.py 는 estimate_direction 을 `chunk=None` 으로만 호출한다.
하지만 파이프라인([pipeline/runner.py])은 실제 4채널 AudioChunk 를 넘기고,
그 출력(DirectionResult)을 FusedResult 에 그대로 끼워 넣는다.

여기서는 그 통합 지점을 검증한다:
  · 실물 청크(모노/4채널)가 들어와도 예외 없이 계약대로 동작하는가
  · 반환 타입이 항상 DirectionResult 계약을 만족하는가
  · DoA 출력이 FusedResult 에 그대로 들어가 to_korean() 까지 동작하는가

ReSpeaker 미연결(노트북) 상황을 기본 가정 → 폴링 경로는 monkeypatch 로 격리.
실행: 레포 루트에서 `pytest tests/test_integration_contract.py`
"""

import numpy as np
import pytest

from core.types import (
    AudioChunk,
    SAMPLE_RATE,
    Direction,
    DirectionResult,
    ClassResult,
    SoundClass,
    ApproachResult,
    Motion,
    FusedResult,
)
from doa import estimator
from doa.estimator import estimate_direction, REAR_RAW_DEG


def _mono_chunk(seconds: float = 1.0) -> AudioChunk:
    """(n,) 모노 청크."""
    n = int(SAMPLE_RATE * seconds)
    return AudioChunk(samples=np.zeros(n, dtype=np.float32))


def _respeaker_chunk(seconds: float = 1.0, channels: int = 4) -> AudioChunk:
    """ReSpeaker 원본처럼 (n, 4) 다채널 청크."""
    n = int(SAMPLE_RATE * seconds)
    rng = np.random.default_rng(0)
    return AudioChunk(samples=rng.standard_normal((n, channels)).astype(np.float32))


@pytest.fixture(autouse=True)
def _no_hardware(monkeypatch):
    """모든 테스트에서 ReSpeaker 미연결을 기본 가정 (폴링 → None)."""
    monkeypatch.setattr(estimator, "_read_respeaker_angle", lambda: None)


def test_real_chunk_no_hardware_returns_unknown():
    """실제 다채널 청크가 들어와도, 하드웨어가 없으면 예외 없이 UNKNOWN."""
    res = estimate_direction(_respeaker_chunk())
    assert res.direction is Direction.UNKNOWN
    assert res.angle_deg is None


def test_return_type_contract():
    """반환은 항상 DirectionResult — direction 은 Direction, angle_deg 는 float|None."""
    for chunk in (_mono_chunk(), _respeaker_chunk()):
        res = estimate_direction(chunk, angle_deg=REAR_RAW_DEG)
        assert isinstance(res, DirectionResult)
        assert isinstance(res.direction, Direction)
        assert res.angle_deg is None or isinstance(res.angle_deg, float)


def test_injected_angle_with_real_chunk():
    """실물 청크 + 주입 각도 → 정상 방향 + 원시 각도 보존."""
    res = estimate_direction(_respeaker_chunk(), angle_deg=REAR_RAW_DEG)
    assert res.direction is Direction.REAR     # 케이블=후방 (보정값 무관 불변)
    assert res.angle_deg == pytest.approx(REAR_RAW_DEG)


def test_output_plugs_into_fused_result():
    """DoA 출력이 FusedResult 에 그대로 들어가 to_korean() 까지 동작 (파이프라인 계약)."""
    direction = estimate_direction(_respeaker_chunk(), angle_deg=REAR_RAW_DEG)
    fused = FusedResult(
        sound=ClassResult.from_label(SoundClass.SIREN, 0.9),
        direction=direction,
        approach=ApproachResult(motion=Motion.APPROACHING),
    )
    text = fused.to_korean()
    assert isinstance(text, str) and text
    assert "후방" in text          # REAR → 후방


def test_chunk_content_does_not_affect_stage1():
    """1단계(MVP)에선 chunk 내용은 결과에 영향 없음 — 각도만이 방향을 결정."""
    a = estimate_direction(_respeaker_chunk(seconds=1.0), angle_deg=90.0)
    b = estimate_direction(_mono_chunk(seconds=0.5), angle_deg=90.0)
    assert a.direction is b.direction
