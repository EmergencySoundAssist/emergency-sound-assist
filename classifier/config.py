"""
classifier 설정 모음 (프로젝트의 '리모컨').

경로, 라벨 매핑, 전처리/학습 하이퍼파라미터, 실험 목록(EXPERIMENTS)을
여기 한곳에 모아둔다. 값 바꿀 일 있으면 이 파일만 고치면 됨.

core/types.py 에 이미 정의된 공통 상수(SoundClass, SAMPLE_RATE, CHUNK_SECONDS)는
중복 정의하지 않고 그대로 가져와 쓴다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from core.types import SoundClass, SAMPLE_RATE, CHUNK_SECONDS

# ---------------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------------
# classifier/config.py 기준으로 두 단계 위가 프로젝트 루트
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 데이터셋 위치 (사용자가 data/UrbanSound8K 에 압축 해제)
DATA_ROOT = PROJECT_ROOT / "data" / "UrbanSound8K"
METADATA_CSV = DATA_ROOT / "metadata" / "UrbanSound8K.csv"
AUDIO_DIR = DATA_ROOT / "audio"                # 안에 fold1 ~ fold10

# 학습 산출물 저장 위치 (.gitignore 처리됨)
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULT_DIR = PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------------
# 라벨 매핑 (UrbanSound8K 10클래스 → 우리 3클래스)
# ---------------------------------------------------------------------------
# CSV의 'class' 문자열 → SoundClass. 여기 없는 클래스는 학습에서 '제외'.
LABEL_MAP: dict[str, SoundClass] = {
    "siren": SoundClass.SIREN,
    "car_horn": SoundClass.HORN,
    "engine_idling": SoundClass.NORMAL_TRAFFIC,
    "street_music": SoundClass.NORMAL_TRAFFIC,
    "air_conditioner": SoundClass.NORMAL_TRAFFIC,
}
# 제외(MVP): children_playing, dog_bark, drilling, jackhammer, gun_shot

# 모델 출력 인덱스 순서 (0,1,2)
CLASSES: list[SoundClass] = [
    SoundClass.SIREN,
    SoundClass.HORN,
    SoundClass.NORMAL_TRAFFIC,
]
CLASS_TO_IDX: dict[SoundClass, int] = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS: dict[int, SoundClass] = {i: c for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

# ---------------------------------------------------------------------------
# 전처리 (Mel-spectrogram)
# ---------------------------------------------------------------------------
# SAMPLE_RATE, CHUNK_SECONDS 는 core/types.py 값 재사용 (16000, 1.0)
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SECONDS)   # 1초 = 16000 샘플
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 64

# ---------------------------------------------------------------------------
# 데이터 분할 (UrbanSound8K 공식 10-fold) — 3분할
#   train(학습) / val(학습 중 선택: early stop·best·LR) / test(최종 1회 평가)
# ---------------------------------------------------------------------------
TRAIN_FOLDS = [1, 2, 3, 4, 5, 6, 7, 8]
VAL_FOLDS = [9]
TEST_FOLDS = [10]

# ---------------------------------------------------------------------------
# 학습 하이퍼파라미터 (모든 실험 공통 — 공정 비교)
# ---------------------------------------------------------------------------
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4             # AdamW 가중치 감쇠
USE_CLASS_WEIGHTS = True        # horn 적음 → 불균형 보정
NUM_WORKERS = 0                 # Windows에서 0이 안전
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 스케줄러 (ReduceLROnPlateau): val_loss 정체 시 LR 감소
SCHED_FACTOR = 0.5
SCHED_PATIENCE = 3
# Early stopping: 검증 macro-F1이 N epoch 동안 안 좋아지면 중단
EARLY_STOP_PATIENCE = 6

# 스모크 테스트 (코드가 끝까지 도는지 빠르게 확인)
SMOKE_EPOCHS = 2
SMOKE_MAX_SAMPLES = 200         # 클래스 무관 총 샘플 제한

# ---------------------------------------------------------------------------
# 실험 정의 (config-driven)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExperimentConfig:
    name: str                   # 체크포인트/결과 식별 이름
    backbone: str               # "scratch" | "yamnet" | "mobilenet"
    head: str                   # "linear" | "mlp"
    freeze_mode: str = "finetune"   # "frozen"(백본 고정) | "finetune"(전체 학습)


# 현재 활성 실험 = 5개
#   비교축: scratch↔pretrained / yamnet↔mobilenet / linear↔mlp
# 주의: YAMNet은 tfhub 구조상 전체 finetune이 어려워 frozen(동결)+head 로 고정.
#       (scratch/mobilenet은 전체학습)
EXPERIMENTS: list[ExperimentConfig] = [
    ExperimentConfig("scratch_cnn",               "scratch",   "mlp",    "finetune"),
    ExperimentConfig("yamnet_frozen_linear",      "yamnet",    "linear", "frozen"),
    ExperimentConfig("yamnet_frozen_mlp",         "yamnet",    "mlp",    "frozen"),
    ExperimentConfig("mobilenet_finetune_linear", "mobilenet", "linear", "finetune"),
    ExperimentConfig("mobilenet_finetune_mlp",    "mobilenet", "mlp",    "finetune"),
]

# 나중에 MobileNet 동결 실험도 비교하고 싶으면 추가:
#   ExperimentConfig("mobilenet_frozen_mlp", "mobilenet", "mlp", "frozen"),
