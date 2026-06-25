# 설계 문서 (docs)

EmergencySoundAssist 설계 문서 목차.

## 공용
- [전체 구조 / 파이프라인](architecture.md)
- [데이터 인터페이스 (팀 약속)](interfaces.md)
- [하드웨어 (Jetson · ReSpeaker 6채널)](hardware.md)
- [용어집](glossary.md)

## 모듈별
- **② 방향 추정 (doa)** — 담당: 천자민
  - [설계](doa/design.md) · [실행 명령어](doa/running.md) · [4방향 매핑/보정](doa/direction-mapping.md) · [다중 음원](doa/multi-source.md) · [Jetson 배포](doa/jetson.md)
- [① 소리 분류 (classifier)](classifier/design.md) — 담당: 이석우·김달현
- [③ 접근/멀어짐 (approach)](approach/design.md) — 담당: 김도윤

---
규칙: **모듈 전용 문서 = 하위폴더 / 여러 모듈 공통 = docs/ 루트.**
