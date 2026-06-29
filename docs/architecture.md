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
- HUD 이전, 파이프라인 콘솔 출력까지 (구현됨 — `python main.py --demo`)
