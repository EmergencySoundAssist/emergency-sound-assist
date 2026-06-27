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
| `spectrum_confidence()` | 주엽 우월도(주엽÷반대편 반원 최대) — 신뢰도 게이팅 지표 | numpy |
| `drop_opposite_phantom()` | primary 기준 ~180° 반대편 가짜 2번째(전후 유령) 제거 | numpy |
| `estimate_multiple_directions()` | 위를 묶어 `[(각도, Direction)]` (`with_confidence=True` 면 `(results, conf)`) | pyroomacoustics |
| `capture_raw4()` | ReSpeaker ch1~4 캡처 | sounddevice |

`pick_peaks`/`mic_locations`/`spectrum_confidence`는 순수 numpy/scipy → **하드웨어·pra 없이 단위 테스트됨**
([tests/test_multi_source.py](../../tests/test_multi_source.py), 16개).

### 신뢰도 게이팅 (`spectrum_confidence`)
작은 4-mic 어레이는 주엽이 넓어 `peak/평균` 이 1 부근에 몰려 둔감하다. 대신 **주엽 중심 ±90°를 뺀
반대편 반원의 최대치**와 비교한다(주엽÷반대편). 한 방향만 뚜렷하면 크고(넓어도), ±180° 반사로
반대편에 맞먹는 봉우리가 생기면 1 부근 → `config.CONF_MIN` 미만이면 실시간 루프가 '방향 불확실' 처리.
시간 다수결과 함께 ±180° 튐을 억제한다([tracking.py](../../doa/tracking.py), [running.md](./running.md)).

## peak 개수 조절 (3개 파라미터)
| 파라미터 | 역할 |
|---|---|
| `height_ratio` | 최대치 대비 임계 (작은 봉우리 제거) |
| `min_sep_deg` | 봉우리 최소 각도 간격 (붙은 것 병합 방지) |
| `max_src` | 최대 음원 수 (보통 2) |

## 설치
```bash
pip install -r requirements.txt   # numpy, scipy, sounddevice, pyroomacoustics, pyusb
```

## 한계 (4-mic 소형 어레이)
- 분리 가능한 동시 음원은 **현실적으로 ~2개**. 그 이상은 신뢰도 급락. 같은 대역(사이렌+사이렌)은 분리 거의 불가, 음색이 다르면(사이렌+경적) 유리.
- 단일 음원에 `--num 2`를 두면 **±180° 유령 2번째 peak**가 뜨는데, `drop_opposite_phantom`
  (config.PHANTOM_TOL_DEG)이 자동 제거 → 다중을 켜도 단일은 1개로 깔끔. MVP 기본은 신뢰성 위해 `config.NUM_SRC=1`.
  기본을 다중으로 하려면 `NUM_SRC=2 + ALGO="MUSIC"` 두 줄.
- 두 소리가 **각도/주파수로 충분히 떨어져야** 분리됨.
- 연산량↑ → pyroomacoustics 는 ARM(Jetson)에서 빌드 무거움 ([jetson.md](./jetson.md)).

## ⚠️ 보정 (검증 순서)
pyroomacoustics 방위각(반시계, 0°=+x)과 ReSpeaker raw DOA 규약이 다를 수 있다.
1. **단일 음원**으로 먼저: 한 방향에서만 소리 → peak 1개가 맞는 방향에 뜨는지
2. 맞으면 **두 방향 동시** → peak 2개 확인
3. `height_ratio`/`min_sep_deg` 튜닝

각도 → 차량 기준(케이블=후방) 매핑은 [direction-mapping.md](./direction-mapping.md)의
보정 상수를 따른다.

## 진행 상태
- [x] `pick_peaks` / `mic_locations` 구현 + 테스트
- [x] `spatial_spectrum` / `estimate_multiple_directions` (pyroomacoustics)
- [x] `spectrum_confidence` 신뢰도 게이팅 + 시간 다수결([tracking.py](../../doa/tracking.py))
- [x] Mac ReSpeaker 실물 단일·다중 방향 + LED 확인
- [ ] `MIC_RADIUS_M` 데이터시트/PCB로 확정 (현재 0.046m 추정)
- [ ] `CONF_MIN`(현재 2.0) 차량/실차 환경 conf 분포로 미세조정
- [ ] Jetson 실물 검증 ([jetson.md](./jetson.md))
