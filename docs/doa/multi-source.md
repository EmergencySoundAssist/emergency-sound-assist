# 다중 음원 방향 추정 (2단계) — `doa/multi_source.py`

> 동시에 여러 방향을 추정한다 (예: 두 대 이상 긴급차 동시 접근). WBS 7주차 고도화 / 스트레치.
> 1단계([estimator.py](../../doa/estimator.py))는 단일 방향만 → 본 모듈은 raw 4채널로 동시 2개+.

---

## 왜 별도 모듈인가
ReSpeaker 내장 DoA(`Tuning.direction`)는 **가장 센 소리 하나**만 각도로 준다. 동시 다방향은
원리상 불가. 그래서 **raw 4채널(ch1~4)을 직접 처리**해 공간 스펙트럼의 봉우리를 여러 개 뽑는다.

## 파이프라인
```
raw 4채널 → STFT → SRP-PHAT 공간 스펙트럼 → 다중 peak 추출 → angle_to_direction
            (채널별)   (사이렌 대역만)        (pick_peaks)      (4방향 매핑)
```

| 함수 | 역할 | 의존성 |
|---|---|---|
| `mic_locations()` | 마이크 좌표(2,4) — 보드 45/135/225/315° | numpy |
| `spatial_spectrum()` | raw → (방위각 그리드, 에너지) | **pyroomacoustics** |
| `pick_peaks()` | 스펙트럼에서 봉우리 여러 개 | numpy/scipy |
| `estimate_multiple_directions()` | 위 둘 묶어 `[(각도, Direction)]` | pyroomacoustics |
| `capture_raw4()` | ReSpeaker ch1~4 캡처 | sounddevice |

`pick_peaks`/`mic_locations`는 순수 numpy → **하드웨어·pra 없이 단위 테스트됨**
([tests/test_multi_source.py](../../tests/test_multi_source.py), 9개).

## peak 개수 조절 (3개 파라미터)
| 파라미터 | 역할 |
|---|---|
| `height_ratio` | 최대치 대비 임계 (작은 봉우리 제거) |
| `min_sep_deg` | 봉우리 최소 각도 간격 (붙은 것 병합 방지) |
| `max_src` | 최대 음원 수 (보통 2) |

## 설치
```bash
pip install pyroomacoustics scipy sounddevice numpy
```

## 한계 (4-mic 소형 어레이)
- 분리 가능한 동시 음원은 **현실적으로 ~2개**. 그 이상은 신뢰도 급락.
- 두 소리가 **각도/주파수로 충분히 떨어져야** 분리됨.
- 연산량↑ → Jetson 단계에서 현실적.

## ⚠️ 보정 (검증 순서)
pyroomacoustics 방위각(반시계, 0°=+x)과 ReSpeaker raw DOA 규약이 다를 수 있다.
1. **단일 음원**으로 먼저: 한 방향에서만 소리 → peak 1개가 맞는 방향에 뜨는지
2. 맞으면 **두 방향 동시** → peak 2개 확인
3. `height_ratio`/`min_sep_deg` 튜닝

각도 → 차량 기준(케이블=후방) 매핑은 [direction-mapping.md](./direction-mapping.md)의
보정 상수를 따른다.

## 진행 상태
- [x] `pick_peaks` / `mic_locations` 구현 + 테스트
- [x] `spatial_spectrum` / `estimate_multiple_directions` 골격 (pyroomacoustics)
- [ ] `MIC_RADIUS_M` 데이터시트/PCB로 확정 (현재 0.046m 추정)
- [ ] 실물 단일→다중 음원 검증 + 파라미터 튜닝
