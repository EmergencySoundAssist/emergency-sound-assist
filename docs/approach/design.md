# 접근 / 멀어짐 판단 설계  *(팀원 담당 — 뼈대)*

> 목표: 시간에 따른 오디오 흐름 → 접근/멀어짐/유지 → `core.types.ApproachResult` 반환
> 구현 파일: [`approach/detector.py`](../../approach/detector.py) — `ApproachDetector.update(chunk) → ApproachResult`

---

## 핵심 아이디어
사이렌이 들렸을 때, 그 차량이 **다가오는지 멀어지는지** 판단.

- **도플러 효과**: 음원이 접근하면 주파수↑, 멀어지면 주파수↓.
- **음량(에너지) 변화**: 접근하면 점점 커지고, 멀어지면 작아짐.

| 관찰 | 판단 |
|------|------|
| 주파수 ↑ & 음량 ↑ | **접근(approaching)** |
| 주파수 ↓ & 음량 ↓ | **멀어짐(receding)** |
| 큰 변화 없음 | 유지(steady) |

---

## 구현 메모
- **상태(이전 청크들)를 기억**해야 추세를 봄 → 클래스(`ApproachDetector`)로 구현.
- 사이렌일 때만 의미 있음 → classifier가 siren일 때만 돌리면 효율적.
- 주파수 추세: 대표 주파수(스펙트럼 피크) 추적 / 음량 추세: RMS 에너지 추적.

## 참고
- 출력 형식: [../interfaces.md](../interfaces.md)
- 용어(도플러): [../glossary.md](../glossary.md)

## TODO (팀원)
- [ ] 청크별 대표 주파수·RMS 음량 계산
- [ ] 시간에 따른 추세로 접근/멀어짐 판단
- [ ] 노트북 단계: 사이렌 음원 접근/후퇴 시뮬레이션으로 검증
