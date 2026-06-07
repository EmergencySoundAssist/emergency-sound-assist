# 방향 추정 (DoA) 설계  *(팀원 담당 — 뼈대)*

> 목표: 오디오 → 전/후/좌/우 4방향 → `core.types.DirectionResult` 반환
> 구현 파일: [`doa/estimator.py`](../../doa/estimator.py) — `estimate_direction(chunk) → DirectionResult`

---

## 2단계 전략

### 1단계 (MVP): ReSpeaker 자체 DoA
- ReSpeaker가 펌웨어에서 계산한 방향 값을 **USB로 읽기만** 함.
- `pyusb` + `tuning.py` → `Tuning.direction` (0~359°)
- 각도 → 4방향 변환 (전 315~45° / 우 45~135° / 후 135~225° / 좌 225~315°)
- 장점: 코드 짧음, DSP 불필요. 한계: 음성용 튜닝이라 도로/사이렌 정확도 미보장.

### 2단계 (개선, 필요시): 4채널 GCC-PHAT / TDOA
- 원본 4채널(ch1~4)로 **마이크 간 도착 시간 차이(TDOA)** 계산 → 방향.
- **GCC-PHAT**로 시간차를 노이즈에 강하게 측정.
- 사이렌 대역(~500~1500Hz) band-pass 후 계산하면 도로 노이즈에 강해짐.
- 라이브러리: `pyroomacoustics`(DoA 알고리즘 내장) 또는 numpy/scipy 직접 구현.

> ⚠️ **먼저 1단계 측정 → 부족하면 2단계.** (과잉설계 방지)

---

## 참고
- 6채널 구성·자체 DoA 읽기: [../hardware.md](../hardware.md)
- 출력 형식: [../interfaces.md](../interfaces.md)
- 평가: 알려진 방향(전/후/좌/우)에서 사이렌 재생 → 정확도 표.

## TODO (팀원)
- [ ] 자체 DoA 값 읽기 + 4방향 변환
- [ ] 노트북 단계: 가짜 각도로 인터페이스 테스트
- [ ] (필요시) GCC-PHAT 구현 + 사이렌 대역 필터
