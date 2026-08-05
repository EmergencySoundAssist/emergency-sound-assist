# 전체 구조와 파이프라인

EmergencySoundAssist는 주변 사이렌·경적을 감지하고 차종, 방향, 접근·멀어짐을 시각 정보로
표시한다. 평상시에는 주변 사람의 말을 STT 자막으로 보여준다.

## 입력 채널

```text
ReSpeaker USB 6채널
  ├─ ch0 처리 음성 → 분류·차종·접근 증거·STT
  ├─ ch1~4 raw    → SRP-PHAT/MUSIC 방향
  └─ ch5 재생 참조 → 현재 미사용
```

물리 마이크는 4개지만 USB 오디오 스트림은 6채널이다. 통합 실행은 반드시
`python main.py --mic --channels 6`으로 연다.

## 런타임 흐름

```text
0.15초 입력 tick
   ↓
2초 사이렌 모델 ─────────────→ PRE 예비경보
5초 사이렌·경적 모델 ───────→ 확정 경보 상태기계
                                   ├─ 사이렌: 차종 6초 다수결
                                   ├─ 사이렌: 접근·멀어짐 조건부 융합
                                   ├─ 긴급음: ch1~4 방향 추정
                                   └─ 긴급 진입: STT 중단·이전 자막 폐기

평상시 ─→ ch0를 1초씩 STT 워커로 전달 ─→ 자막
```

경보 상태기계는 ONSET, 지속 리마인더, CLEAR를 만들며 짧은 단일 예측보다 안정적으로
켜고 끈다. 접근·멀어짐은 `speed_neural_dir`, 음량 추세, 직접 도플러를 고정 평균하지 않고
조건에 따라 선택한다. 검증되지 않은 km/h와 느림/보통/빠름은 표시하지 않는다.

STT는 백그라운드에서 실행되므로 Whisper가 느려도 긴급음 검출 루프는 계속 돈다. 긴급 진입
시 세대 번호를 바꿔 대기 중·변환 중·완료된 이전 자막을 모두 무효화한다.

## 모듈별 책임

| 모듈 | 경로 | 입력 | 역할 |
|---|---|---|---|
| 분류·차종 | `classifier/` | ch0 | siren/horn/noise, PRE, 구급/경찰/소방 |
| 방향 | `doa/` | ch1~4 | 전/후/좌/우 |
| 접근 | `approach/`, `pipeline/motion_fusion.py` | ch0 | 접근/멀어짐/유지/미상 |
| STT | `stt/` | ch0 | 평상시 한국어 자막 |
| 통합 | `pipeline/` | 위 결과 | 경보 상태와 화면용 정보 생성 |

## 실제 반환값

현재 실시간 `Pipeline.process(AudioChunk)`는 다음을 반환한다.

```text
(AlertEvent, info)
```

- `AlertEvent`: NONE/PRE/WARN/CRITICAL, 종류, 차종, 위험 문구, onset/remind/clear
- `info`: 방향, 움직임, 근접 게이지, 모델·음량·도플러 진단값, STT 상태 등

세부 타입과 `info` 키는 [인터페이스](interfaces.md)를 참고한다.

## 연산 장치

| 처리 | 실행 경로 |
|---|---|
| 검출·차종·움직임 ONNX | TensorRT → CUDA → CPU provider 우선순위 |
| faster-whisper STT | CTranslate2 CUDA 또는 CPU/int8 |
| 음량·직접 도플러·방향 후처리 | CPU |

TensorRT와 CTranslate2는 같은 Jetson GPU를 공유한다. TensorRT가 활성화돼도 STT가 자동으로
GPU를 사용하는 것은 아니므로 각각 확인해야 한다. 명령은 [실행 문서](running.md#8-jetson-gpu--tensorrt-확인)에 있다.

## 현재 검증 범위

- 코드 회귀 테스트와 공개 데이터/물리 시뮬레이션 검증 완료
- 실제 ReSpeaker 차량 내부, 차체 반사·바람, Jetson 동시 GPU 부하는 최종 검증 전
- 접근 속도 단계와 절대 km/h는 출력하지 않음
- HUD 하드웨어 연결 전이며 콘솔·대시보드 출력까지 구현
