# 방향 추정 (DoA) 설계  *(담당: 천자민)*

> 목표: 오디오 → 전/후/좌/우 4방향 → `core.types.DirectionResult` 반환
> 구현 파일: [`doa/estimator.py`](../../doa/estimator.py) — `estimate_direction(chunk) → DirectionResult`
> 실행/옵션은 [running.md](./running.md), 다중 음원은 [multi-source.md](./multi-source.md), Jetson은 [jetson.md](./jetson.md).

---

## 두 실행 경로

### 경량 경로: ReSpeaker 자체 DoA
- ReSpeaker가 펌웨어에서 계산한 방향 값을 **USB로 읽기만** 함.
- `pyusb` + [`respeaker_tuning.py`](../../doa/respeaker_tuning.py) → `Tuning.direction` (0~359°)
- 각도 → 4방향 변환([estimator.py](../../doa/estimator.py) `angle_to_direction`), 케이블=후방 보정([direction-mapping.md](./direction-mapping.md)).
- 경량(numpy+pyusb)이라 Jetson 최초 브링업 경로([`doa/live.py`](../../doa/live.py)).

### 통합 주 경로: 4채널 SRP-PHAT / MUSIC
- 원본 4채널(ch1~4)을 직접 처리해 **공간 스펙트럼**에서 동시 여러 방향 추정([multi_source.py](../../doa/multi_source.py)).
- `pyroomacoustics` 의 **SRP-PHAT**(기본, 가벼움) / **MUSIC**(다중 분리 유리). 사이렌 대역(~500~1500Hz)만 사용.
  > 초기 계획은 GCC-PHAT 직접 구현이었으나, pyroomacoustics 의 SRP/MUSIC 로 대체(다중 음원·유지보수 이점).
- **견고화**: 시간 다수결([tracking.py](../../doa/tracking.py)) + 신뢰도 게이팅(`spectrum_confidence`)으로
  반사/터널의 ±180° 튐을 억제. 실시간 루프 [`multi_live.py`](../../doa/multi_live.py).

`main.py --mic --channels 6`은 두 번째 오디오 경로의 SRP-PHAT을 사용한다. 자체 DoA 경로는
장치 브링업과 가벼운 단독 실행용으로 유지한다.

---

## 참고
- 6채널 구성·자체 DoA 읽기: [../hardware.md](../hardware.md)
- 출력 형식: [../interfaces.md](../interfaces.md)
- 평가: 알려진 방향(전/후/좌/우)에서 사이렌 재생 → 정확도 표 (진단 도구 `python -m doa.diag`).

## TODO
- [x] 자체 DoA 값 읽기 + 4방향 변환 (1단계)
- [x] 다중 음원 SRP/MUSIC + 실시간 루프 + LED (2단계)
- [x] 시간 다수결 + 신뢰도 게이팅(±180° 튐 억제) + 진단 도구
- [ ] 차량 장착 후 보정값 실측 확정 (`REAR_RAW_DEG`/`MIRROR`/`LED_OFFSET`/`MIC_RADIUS_M`/`CONF_MIN`)
- [ ] Jetson 실물 검증 + 4방향 정확도 평가(혼동행렬)
