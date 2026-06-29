# EmergencySoundAssist

청각장애 운전자를 위한 **긴급 경고음 감지 보조 디바이스**.
차량 주변의 사이렌·경적 같은 소리를 AI로 감지해, **종류·방향·접근 여부**를 시각 정보로 바꿔준다.

> 최종 출력 예: **"구급차, 후방, 빠르게 접근 중"**  (사이렌이면 차종·속도까지)

---

## MVP 3대 기능
| # | 기능 | 내용 | 담당 |
|---|------|------|------|
| ① | 소리 분류 | siren / horn / normal + 차종 + 접근속도(1~5) | 이석우, 김달현 |
| ② | 방향 추정 | 전 / 후 / 좌 / 우 | 천자민 |
| ③ | 접근/멀어짐 | 도플러 + 음량 추세 | 김도윤 |

> **확장(④ STT)**: 주변 음성(확성기·외침·안내방송)을 **텍스트로** 바꿔 청각장애 운전자에게 보여 준다. → [docs/stt/design.md](docs/stt/design.md) · 담당: 천자민

---

## 프로젝트 구조
```
core/         공통 데이터 약속 (types.py)
audio/        오디오 입력 (마이크/파일)
classifier/   ① 소리 분류  ← 구현됨 (CNN+Attn ONNX 검출 + 사이렌 차종)
doa/          ② 방향 추정  ← 구현됨 (자체 DoA·다중음원 SRP/MUSIC·LED·스무딩·진단)
approach/     ③ 접근/멀어짐 ← 구현됨 (실시간 도플러 + 음량 추세)
stt/          ④ STT 음성→텍스트 ← 구현됨 (faster-whisper · 확장, 독립 실행)
pipeline/     세 결과 통합  ← 구현됨 (FusedResult → "구급차, 후방, 접근 중")
docs/         설계 문서  → docs/README.md
```

## 설치 / 실행
```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -U pip
pip install -r requirements.txt       # 의존성 설치

# ① 분류 + 차종
python classify.py --demo             # 합성 사이렌으로 분류+차종 확인
python classify.py --mic              # 실시간 마이크

# 통합 파이프라인 (분류+차종+접근+방향)
python main.py --demo                          # 합성 (방향은 미상 — 1채널)
python main.py --mic --channels 6 --device 0   # ReSpeaker → 4개 전부 한 줄로
python main.py --mic --channels 6 --device 0 --stt --stt-model tiny  # + 평상시 자막 (노트북은 tiny 권장)
#   STT 는 백그라운드 스레드로 돌아 메인을 안 막음 / 사이렌·경적일 땐 자동 멈춤 / faster-whisper 필요

# ② 방향만 단독 (ReSpeaker 필요)
python -m doa.multi_live --led        # 실시간 방향 + LED
python -m pytest -q                   # 테스트
```
실행 옵션 → [docs/doa/running.md](docs/doa/running.md) · Jetson 배포 → [docs/doa/jetson.md](docs/doa/jetson.md)

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
- [① 분류](docs/classifier/design.md) · [↳ 차종](docs/classifier/subtype.md) · [② 방향](docs/doa/design.md) · [③ 접근](docs/approach/design.md) · [④ STT](docs/stt/design.md)
