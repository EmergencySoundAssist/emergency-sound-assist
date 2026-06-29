# 전체 구조 / 파이프라인

청각장애 운전자용 보조 디바이스. 차량 주변 소리를 시각 정보로 바꿔준다.

---

## 전체 흐름
```
[ReSpeaker 마이크 6채널]
        │
        ├── ch0 (처리됨)  ──→  ① 분류 (classifier)   → ClassResult   (siren/horn/normal)
        │                          └─ siren 이면 차종 추론 → 구급/경찰/소방 (subtype)
        │
        └── ch1~4 (원본)  ──→  ② 방향 (doa)          → DirectionResult (전/후/좌/우)
                          └─→  ③ 접근 (approach)      → ApproachResult  (접근/멀어짐)
        │
        ▼
   [pipeline] 세 결과 통합 → FusedResult
        ▼
   출력 예: "구급차, 후방, 접근 중"   (→ 이후 HUD/디스플레이)
```

> ① 분류는 `siren` 일 때 한 단계 더 들어가 차종(구급/경찰/소방)을 붙인다. → [classifier/subtype.md](classifier/subtype.md)
> 방향은 ReSpeaker 6채널 한 스트림을 ch0(분류·접근)·ch1~4(SRP-PHAT 방향)로 갈라 `pipeline` 이 통합한다 (`python main.py --mic --channels 6`). ReSpeaker 없으면 방향만 미상.
> ④ STT 는 **분류가 게이트**한다 — 사이렌·경적(긴급)이면 STT 를 멈추고, 평상시(noise)면 자막을 만든다 (`--stt`, WBS 10주차 우선순위 전환).

---

## 세 모듈 (담당 분리)
| # | 모듈 | 폴더 | 설계 문서 | 담당 |
|---|------|------|----------|------|
| ① | 소리 분류 (+차종) | `classifier/` | [design](classifier/design.md) · [subtype](classifier/subtype.md) | 이석우·김달현 |
| ② | 방향 추정 | `doa/` | [doa/design.md](doa/design.md) | 천자민 |
| ③ | 접근/멀어짐 | `approach/` | [approach/design.md](approach/design.md) | 김도윤 |
| ④ | STT 음성→텍스트 | `stt/` | [stt/design.md](stt/design.md) | 천자민 *(MVP 외 확장)* |
| - | 통합 | `pipeline/` | - | 공통 |

- 세 모듈은 **`core/types.py`의 데이터 약속**으로 연결 → [interfaces.md](interfaces.md)

---

## 개발 단계
1. **노트북(CPU)** 에서 각 모듈 개발/검증 (현재)
2. **Jetson Orin Nano + ReSpeaker** 로 이식 → 실시간 통합
3. (이후) HUD/디스플레이 연결

---

## MVP 범위
- 분류: siren / horn / normal_traffic (3클래스) — CNN+Attn ONNX, 구현됨
  - +차종: siren 일 때 구급/경찰/소방 (선택 — 모델 없으면 자동 생략)
- 방향: 4방향 — 자체 DoA(1단계) + 다중음원 SRP-PHAT/MUSIC(2단계), 구현됨
- 접근: 도플러 + 음량 추세, 구현됨
- ④ STT(확장): 평상시 음성→자막. 분류가 긴급/평상시 게이트 (`--stt`)
- HUD 이전, 파이프라인 콘솔 출력까지 (구현됨 — `python main.py --demo`)

---

## STT 게이트 (긴급/평상시 전환 — 1단계)

분류를 '스위치'로 써서 STT 를 켜고 끈다 (WBS 10주차 '긴급↔STT 우선순위 자동 전환').

| 소리 | 분류 | STT | 출력 |
|------|------|-----|------|
| 사이렌·경적 | siren/horn (긴급) | 멈춤 | 차종 + 방향 + 접근 경고 |
| 사람 말 | noise | 돌림 → 텍스트 | 자막 |
| 그 외 소음 | noise | 돌림 → STT 가 거름 | (자막 없음) |

- **1단계(현재)**: "사이렌·경적만 아니면 STT 허용". 말소리는 분류상 `noise` 라 STT 로 흐른다.
  사람 말이 아닌 소음은 STT 의 **VAD(소리 크기) + 환각 가드**(`no_speech_threshold` 등)가 걸러
  자막이 안 뜬다. 한계: 시끄러운 소음에 엔진을 한 번 돌리는 낭비 + 드문 헛인식.
- **2단계(개선, 필요시)**: 분류에 `speech` 클래스를 추가(siren/horn/**speech**/noise)해
  진짜 말소리만 STT 로 보낸다 → 낭비·헛인식 0. 단 모델 재학습(말소리 데이터)이 필요하다.

> STT 는 **백그라운드 워커 스레드**(`stt/worker.py`)로 돈다 — 인식(수 초 블로킹)이 메인을
> 막지 않아 **STT 중에도 사이렌을 놓치지 않는다**. 메인은 `feed()`/`reset()` 만 호출(즉시 반환),
> 완성된 자막은 `latest()` 로 받아 출력한다. 긴급 진입 시 발화 버퍼는 `reset()` 으로 비운다.
