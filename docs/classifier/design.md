# 소리 분류 (Classifier) 설계

> 담당: 소리 분류 모듈
> 목표: 1초 오디오 청크 → `siren` / `horn` / `normal_traffic` 분류 → `core.types.ClassResult` 반환

---

## 1. 목표와 범위

- 입력: 1초 길이 오디오 (16kHz 모노)
- 출력: `ClassResult(label, confidence, is_emergency)`
- 클래스 3종: `siren`(사이렌) / `horn`(경적) / `normal_traffic`(일반 도로 소음)
- 노트북(CPU)에서 개발 → Jetson Orin Nano로 이식

---

## 2. 데이터셋: UrbanSound8K

- 도시 소음 8,732개 녹음(각 ≤4초), **공식 10-fold 분할** 제공 → train/test 누수 자동 방지
- 라벨은 `metadata/UrbanSound8K.csv` (`slice_file_name`, `fold`, `classID`, `class` 등)

### 라벨 매핑 (10클래스 → 우리 3클래스)

| UrbanSound8K class | classID | → 우리 클래스 |
|--------------------|:------:|--------------|
| `siren` | 8 | **siren** |
| `car_horn` | 1 | **horn** |
| `engine_idling` | 5 | **normal_traffic** |
| `street_music` | 9 | **normal_traffic** |
| `air_conditioner` | 0 | **normal_traffic** |
| 그 외(children_playing, dog_bark, drilling, jackhammer, gun_shot) | - | **제외(MVP)** |

- **normal_traffic = 도로 관련만** (engine/street_music/air_conditioner). 깔끔한 경계 → MVP 정확도·디버깅 유리.
- 확장 시: "그 외도 전부 normal" 또는 사이렌 세분류(sireNNet)로 매핑만 교체.

### 데이터 양 / 주의
- siren ~929, horn ~429, normal ~3,000 → **horn이 가장 적고 불균형(~7×)** → 학습 시 **class weight**로 보정.

---

## 3. 전처리

- 공통: 16kHz 모노, 1초(16,000 샘플)로 crop/pad
- **Mel-spectrogram**(소리를 이미지처럼): `n_fft=1024`, `hop_length=512`, `n_mels=64` → log scale(dB) → 정규화
  - 구현: `torchaudio.transforms.MelSpectrogram` + `AmplitudeToDB`
  - MobileNet 입력용: (1,64,~32) 1채널 → **3채널 복제**
- YAMNet은 자체 전처리(16kHz 파형 그대로 입력 → 내부에서 mel/임베딩 생성)

> 📖 단계별 자세한 설명(실제 예시 포함): [preprocessing.md](preprocessing.md)

---

## 4. 모델 실험 — 전체학습 비교 (시작점 × head)

**모든 활성 실험은 백본 전체를 학습(full training).** 차이는 두 축뿐:

- 축1 **시작점(backbone)**: `scratch`(랜덤) / `yamnet`(오디오 사전학습) / `mobilenet`(이미지 사전학습)
- 축2 **head**: `linear`(1층) / `mlp`(은닉층 포함)

### 활성 실험 5개

| 이름 | 시작점 | head | 학습 | 역할 |
|------|--------|------|------|------|
| `scratch_cnn` | 랜덤 | mlp | 전체 | **베이스라인** |
| `yamnet_frozen_linear` | YAMNet | linear | head만(동결) | |
| `yamnet_frozen_mlp` | YAMNet | mlp | head만(동결) | |
| `mobilenet_finetune_linear` | MobileNet | linear | 전체 | |
| `mobilenet_finetune_mlp` | MobileNet | mlp | 전체 | |

> ⚠️ YAMNet은 tfhub 구조상 전체 finetune이 어려워 **frozen(동결)+head** 로 구현.
> YAMNet(TF)으로 1024 임베딩을 뽑고(동결), 그 위에 torch head만 학습. (scratch/MobileNet은 전체학습)

### 비교로 알고 싶은 것
- **scratch ↔ pretrained**: 사전학습 시작점이 도움 되나?
- **yamnet ↔ mobilenet**: 오디오 vs 이미지 사전학습?
- **linear ↔ mlp**: head를 키우면 좋아지나?

> ⚠️ 전부 백본 전체 학습이라 **CPU에서 느림** → 먼저 epoch 2~3 스모크 후 본학습.
> (확장 여지) 코드에 `freeze_mode` 옵션(동결=head만 학습) 있음 — 지금은 미사용.

### 공통 학습 설정
- 손실: CrossEntropyLoss (**class weight**로 불균형 보정)
- 최적화: **AdamW**(lr=1e-3, weight_decay=1e-4), batch=32, **epochs=60**
- 스케줄러: **ReduceLROnPlateau** (val_loss 정체 시 LR 감소, factor=0.5, patience=3)
- **Early stopping**: val macro-F1이 **10 epoch** 동안 안 좋아지면 중단
- ❌ 이미지 전용(448 입력, ImageNet 정규화, rotation/flip 증강)은 오디오에 안 맞아 미사용

> 설정 위치: `classifier/config.py` (`EPOCHS=60`, `EARLY_STOP_PATIENCE=10`).
> 초기값(epochs 30 / early stop 6)에서 학습을 더 충분히 돌리도록 상향함.

---

## 5. 학습 / 평가 (3분할)

- **3분할**: train = fold 1~8 / **val = fold 9** / test = fold 10
  - val: 학습 중 선택(early stop·best·LR) / test: **최종 1회** 평가(보고용) → 점수 부풀림 방지
- best 체크포인트: **검증(val) macro-F1** 기준 저장
- 지표: 정확도 + 클래스별 precision/recall + **혼동행렬** (모두 test 기준)
- **비교 표**: 5개 설정의 test 정확도·macro-F1 한눈에 → 이긴 모델을 Jetson 메인으로 (`outputs/comparison.md`)

---

## 6. 코드 구조 (`classifier/`)

| 파일 | 역할 |
|------|------|
| `config.py` | 경로, 라벨 매핑, 하이퍼파라미터, **EXPERIMENTS 목록** |
| `preprocessing.py` | 오디오 → log-mel 텐서 |
| `dataset.py` | UrbanSound8K Dataset (3클래스 매핑, fold split) |
| `models/backbones.py` | scratch CNN / MobileNetV3 (freeze_mode 옵션) |
| `models/heads.py` | Linear / MLP head |
| `models/yamnet.py` | YAMNet 임베딩 추출 (TF, 동결) |
| `models/build.py` | 설정 → 모델 조립 |
| `train.py` | EXPERIMENTS 루프 학습 (3분할, `--smoke`/`--exp` 옵션) |
| `evaluate.py` | test 비교 표 → `outputs/comparison.md` |
| `infer.py` | best 모델 → `AudioChunk → ClassResult` |

> 통합 데모: `pipeline/run.py` (`--wav`/`--mic`) → "사이렌, 방향 미상, 이동 미상"

---

## 7. 인터페이스 (팀 약속)

- 입력: `core.types.AudioChunk`
- 출력: `core.types.ClassResult(label: SoundClass, confidence: float, is_emergency: bool)`
- → 자세한 데이터 약속은 [../interfaces.md](../interfaces.md)

---

## 8. 의존성
- torch, torchaudio, torchvision (scratch/MobileNet)
- tensorflow, tensorflow-hub (YAMNet)
- soundfile, pandas, scikit-learn, tqdm
- (Python 3.13에서 tensorflow wheel 호환 확인 필요)

---

## 9. 학습 결과 (2026-06-09, 1차)

- 환경: 데스크탑 GPU (NVIDIA GTX 1050 Ti, CUDA 11.8 / torch 2.7.1+cu118), Windows
- 설정: epochs=60, early stop patience=10, batch=32, 3분할(train 1~8 / val 9 / test 10)
- 명령: `python -m classifier.train` → `python -m classifier.evaluate`

### 비교 표 (test = fold 10)

| 순위 | experiment | acc | macro-F1 | Jetson 이식 |
|:--:|---|:--:|:--:|---|
| 🥇 | **mobilenet_finetune_linear** | 0.878 | **0.839** | ✅ PyTorch만 |
| 🥈 | yamnet_frozen_mlp | 0.890 | 0.837 | ⚠️ TensorFlow 필요 |
| 🥉 | scratch_cnn | 0.883 | 0.832 | ✅ PyTorch만 (가장 가벼움) |
| 4 | yamnet_frozen_linear | 0.878 | 0.825 | ⚠️ TensorFlow 필요 |
| 5 | mobilenet_finetune_mlp | 0.863 | 0.813 | ✅ PyTorch만 |

**채택 메인 모델: `mobilenet_finetune_linear` (macro-F1 0.839)**
- 5개 모델 성능이 macro-F1 0.81~0.84로 비슷 → 큰 차이 없음.
- 1등인데 **PyTorch만으로 추론**돼서 Jetson 이식이 간단(= TF 의존성 없음). YAMNet 계열은 점수 차 0.002뿐인데 TF 부담만 큼.
- 공통 약점: **siren의 recall이 낮음**(전 모델에서 siren을 normal_traffic으로 일부 오분류). 향후 siren 데이터 보강/증강 여지.

> 상세 클래스별 precision/recall·혼동행렬은 `outputs/comparison.md` 참고.
> ⚠️ `outputs/`, `checkpoints/`는 `.gitignore` 처리 → 산출물은 로컬에만 존재(깃 미추적). 이 표는 영구 기록용으로 여기에 복사해 둠.

### Jetson 이식 메모
- 옮길 파일: `checkpoints/mobilenet_finetune_linear.pt` (state_dict 저장 → 하드웨어 독립적).
- 로드 시 `infer.py`가 `map_location=config.DEVICE`로 처리 → Jetson GPU/CPU 자동 매핑.
- `infer.py`는 `checkpoints/`에서 macro-F1 최고 모델을 자동 선택.
- 주의: Jetson에는 **JetPack 버전에 맞는 NVIDIA 제공 PyTorch 휠** 설치 필요(일반 pip torch 불가). MobileNet/scratch는 PyTorch만으로 추론돼 이식이 쉬움.

---

## 10. 환경 / 트러블슈팅 (Windows · 한글 경로)

한국어 Windows + 한글 사용자명(`C:\Users\최병호\...`) 환경에서 겪은 이슈와 해결:

| 증상 | 원인 | 해결 |
|------|------|------|
| `pip install -r requirements.txt` → `UnicodeDecodeError: 'cp949' codec ...` | requirements.txt가 UTF-8(한글 주석·`—`·`→`)인데 pip가 cp949로 디코드 | requirements.txt에 **UTF-8 BOM** 추가 → pip가 UTF-8로 자동 인식 |
| YAMNet 실험에서 `FailedPreconditionError: ...\Temp is not a directory` | TF Hub 캐시 경로에 한글이 섞여 TF 파일 IO가 비ASCII 경로 처리 실패 | `models/yamnet.py`에 **`_ensure_ascii_cache_dir()`** 추가 → 캐시 경로에 비ASCII가 있으면 시스템 드라이브 루트의 영문 폴더(`C:\tfhub_cache`)로 자동 우회. import 시 자동 실행되어 환경변수 수동 설정 불필요 |
| 콘솔 한글 깨짐 | Windows 콘솔 기본 인코딩(cp949) | 실행 전 `$env:PYTHONUTF8=1` (또는 `set PYTHONUTF8=1`) |

> 직접 `TFHUB_CACHE_DIR` 환경변수를 지정하면 그 값이 우선 적용됨(코드가 존중).
