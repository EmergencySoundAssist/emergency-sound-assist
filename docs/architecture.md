# 전체 구조 / 파이프라인

청각장애 운전자용 보조 디바이스. 차량 주변 소리를 시각 정보로 바꿔준다.

---

## 전체 흐름
```
[ReSpeaker 마이크 6채널]
        │
        ├── ch0 (처리됨)  ──→  ① 분류 (classifier)   → ClassResult   (siren/horn/normal)
        │
        └── ch1~4 (원본)  ──→  ② 방향 (doa)          → DirectionResult (전/후/좌/우)
                          └─→  ③ 접근 (approach)      → ApproachResult  (접근/멀어짐)
        │
        ▼
   [pipeline] 세 결과 통합 → FusedResult
        ▼
   출력 예: "사이렌, 후방, 접근 중"   (→ 이후 HUD/디스플레이)
```

---

## 세 모듈 (담당 분리)
| # | 모듈 | 폴더 | 설계 문서 | 담당 |
|---|------|------|----------|------|
| ① | 소리 분류 | `classifier/` | [classifier/design.md](classifier/design.md) | 나 |
| ② | 방향 추정 | `doa/` | [doa/design.md](doa/design.md) | 팀원 |
| ③ | 접근/멀어짐 | `approach/` | [approach/design.md](approach/design.md) | 팀원 |
| - | 통합 | `pipeline/` | - | 공통 |

- 세 모듈은 **`core/types.py`의 데이터 약속**으로 연결 → [interfaces.md](interfaces.md)

---

## 개발 단계
1. **노트북(CPU)** 에서 각 모듈 개발/검증 (현재)
2. **Jetson Orin Nano + ReSpeaker** 로 이식 → 실시간 통합
3. (이후) HUD/디스플레이 연결

---

## MVP 범위
- 분류: siren / horn / normal_traffic (3클래스)
- 방향: 4방향 (자체 DoA → 필요시 GCC-PHAT)
- 접근: 도플러 + 음량 추세
- HUD 이전, 파이프라인 콘솔 출력까지
