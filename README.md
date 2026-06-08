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

---

## 프로젝트 구조
```
core/         공통 데이터 약속 (types.py)
audio/        오디오 입력 (마이크/파일)
classifier/   ① 소리 분류  ← 구현 중
doa/          ② 방향 추정  (스텁)
approach/     ③ 접근/멀어짐 (스텁)
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

## 실행 방법 (classifier)

### 1. 환경 준비
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # (cmd: .venv\Scripts\activate.bat)
pip install -r requirements.txt
```
> Python 3.13에서 전역 TF가 안 깔리면 venv 안에서 설치하세요. (TF 2.21+ 필요)

### 2. 데이터 준비 (UrbanSound8K)
1. [Kaggle](https://www.kaggle.com/datasets/chrisfilo/urbansound8k) 또는 [Zenodo](https://zenodo.org/records/1203745)에서 다운로드
2. 압축 해제 후 아래 구조로 `data/`에 배치:
   ```
   data/UrbanSound8K/metadata/UrbanSound8K.csv
   data/UrbanSound8K/audio/fold1 ... fold10/
   ```

### 3. 학습 / 평가 / 실행
```bash
python -m classifier.train --smoke      # 빠른 검증(2 epoch, 소수 샘플)
python -m classifier.train              # 본학습 (전체, 5개 실험)
python -m classifier.evaluate           # test(fold10) 비교 표 → outputs/comparison.md
python -m pipeline.run --wav <파일>     # 통합 데모 ("사이렌, 방향 미상, 이동 미상")
python -m pipeline.run --mic            # 마이크 실시간
```
> 한글 출력이 깨지면 `set PYTHONUTF8=1` (또는 `$env:PYTHONUTF8=1`) 후 실행.

### 모델 실험 (5개, 전체학습 비교)
`scratch_cnn`(베이스라인) / `yamnet_frozen_{linear,mlp}` / `mobilenet_finetune_{linear,mlp}`
→ 자세히: [docs/classifier/design.md](docs/classifier/design.md)

## 문서
설계 문서는 [`docs/`](docs/README.md) 참고:
- [전체 구조](docs/architecture.md) · [데이터 인터페이스](docs/interfaces.md) · [하드웨어](docs/hardware.md) · [용어집](docs/glossary.md)
- [① 분류](docs/classifier/design.md) · [② 방향](docs/doa/design.md) · [③ 접근](docs/approach/design.md)
