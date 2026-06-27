# EmergencySoundAssist

청각장애 운전자를 위한 **긴급 경고음 감지 보조 디바이스**.
차량 주변의 사이렌·경적 같은 소리를 AI로 감지해, **종류·방향·접근 여부**를 시각 정보로 바꿔준다.

> 최종 출력 예: **"사이렌, 후방, 접근 중"**

---

## MVP 3대 기능
| # | 기능 | 내용 | 담당 |
|---|------|------|------|
| ① | 소리 분류 | siren / horn / normal_traffic | 이석우, 김달현 |
| ② | 방향 추정 | 전 / 후 / 좌 / 우 | 천자민 |
| ③ | 접근/멀어짐 | 도플러 + 음량 추세 | 김도윤 |

> **확장(④ STT)**: 주변 음성(확성기·외침·안내방송)을 텍스트로 바꾸고 긴급 키워드(구급차·비키세요 등)를 짚어 준다. → [docs/stt/design.md](docs/stt/design.md) · 담당: 천자민

---

## 프로젝트 구조
```
core/         공통 데이터 약속 (types.py)
audio/        오디오 입력 (마이크/파일)
classifier/   ① 소리 분류  ← 구현 중
doa/          ② 방향 추정  (스텁)
approach/     ③ 접근/멀어짐 (스텁)
stt/          ④ STT 음성→텍스트 (확장)
pipeline/     세 결과 통합
docs/         설계 문서  → docs/README.md
```

---

## 하드웨어
- NVIDIA Jetson Orin Nano Super Dev Kit
- microSD 128GB
- ReSpeaker USB Mic Array (XVF-3000, 4-mic)

자세히 → [docs/hardware.md](docs/hardware.md)

---

## 개발 단계
1. **노트북(CPU)** 에서 각 모듈 개발/검증 (현재)
2. **Jetson + ReSpeaker** 이식 → 실시간 통합
3. (이후) HUD/디스플레이 연결

---

## 문서
설계 문서는 [`docs/`](docs/README.md) 참고:
- [전체 구조](docs/architecture.md) · [데이터 인터페이스](docs/interfaces.md) · [하드웨어](docs/hardware.md) · [용어집](docs/glossary.md)
- [① 분류](docs/classifier/design.md) · [② 방향](docs/doa/design.md) · [③ 접근](docs/approach/design.md) · [④ STT](docs/stt/design.md)
