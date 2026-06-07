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

---

## 4. 모델 실험 — 전체학습 비교 (시작점 × head)

**모든 활성 실험은 백본 전체를 학습(full training).** 차이는 두 축뿐:

- 축1 **시작점(backbone)**: `scratch`(랜덤) / `yamnet`(오디오 사전학습) / `mobilenet`(이미지 사전학습)
- 축2 **head**: `linear`(1층) / `mlp`(은닉층 포함)

### 활성 실험 5개

| 이름 | 시작점 | head | 역할 |
|------|--------|------|------|
| `scratch_cnn` | 랜덤 | (단순) | **베이스라인** |
| `yamnet_finetune_linear` | YAMNet | linear | |
| `yamnet_finetune_mlp` | YAMNet | mlp | |
| `mobilenet_finetune_linear` | MobileNet | linear | |
| `mobilenet_finetune_mlp` | MobileNet | mlp | |

### 비교로 알고 싶은 것
- **scratch ↔ pretrained**: 사전학습 시작점이 도움 되나?
- **yamnet ↔ mobilenet**: 오디오 vs 이미지 사전학습?
- **linear ↔ mlp**: head를 키우면 좋아지나?

> ⚠️ 전부 백본 전체 학습이라 **CPU에서 느림** → 먼저 epoch 2~3 스모크 후 본학습.
> (확장 여지) 코드에 `freeze_mode` 옵션(동결=head만 학습) 있음 — 지금은 미사용.

### 공통 학습 설정
- 손실: CrossEntropyLoss (**class weight**로 불균형 보정)
- 최적화: Adam(lr=1e-3), batch=32, epochs≈30

---

## 5. 학습 / 평가

- **동일 분할**: train = fold 1~9, test = fold 10 (공정 비교)
- best 체크포인트: **검증 macro-F1** 기준 저장
- 지표: 정확도 + 클래스별 precision/recall + **혼동행렬**
- **비교 표**: 5개 설정의 정확도·macro-F1 한눈에 → 이긴 모델을 Jetson 메인으로

---

## 6. 코드 구조 (`classifier/`)

| 파일 | 역할 |
|------|------|
| `config.py` | 경로, 라벨 매핑, 하이퍼파라미터, **EXPERIMENTS 목록** |
| `preprocessing.py` | 오디오 → log-mel 텐서 |
| `dataset.py` | UrbanSound8K Dataset (3클래스 매핑, fold split) |
| `models/backbones.py` | scratch CNN / YAMNet / MobileNetV3 (freeze_mode 옵션) |
| `models/heads.py` | Linear / MLP head |
| `models/build.py` | 설정 → 모델 조립 |
| `train.py` | EXPERIMENTS 루프 학습 |
| `evaluate.py` | 비교 표 출력 |
| `infer.py` | 메인 모델 → `AudioChunk → ClassResult` |

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
