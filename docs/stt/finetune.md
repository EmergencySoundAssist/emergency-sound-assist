# STT 파인튜닝 — 평가셋 v0 · baseline 측정 기록

설계: `docs/superpowers/specs/2026-07-06-stt-finetune-data-eval-design.md`
실행법: `finetune/README.md`

## 평가셋 v0 (SEED 20260706, 총 350발화)

| 구성 | 소스 | 수 |
|---|---|---|
| 일반 발화 | zeroth-korean test(CC-BY-4.0) 130 + google/fleurs ko_kr test(CC-BY-4.0) 70 | 200 |
| 긴급문구 | 75문장 × edge-tts 2화자(SunHi·InJoon) | 150 |

조건: corpus = clean/snr10/snr5/snr0 각 50. tts = loudspeaker 75(소음 SNR 10/5/0 순환)
+ 나머지 75 를 4조건 순환. 노이즈 = DEMAND TCAR/TBUS/STRAFFIC + MS-SNSD 차량 4종.

## 스펙 편차 (2026-07-06 확정)

- Common Voice ko 제외 — Mozilla 가 2025-10 HF 배포 중단(Data Collective 이관).
- TTS 화자 풀 = edge-tts 한국어 3종뿐. 평가 SunHi+InJoon / 미래 학습 Hyunsu 로 분리.
- UrbanSound8K(6GB) 기본 제외 — 사이렌 겹침 조건 v0 미포함. `data/noise/extra/` drop-in 지원.
- 평가 엔진은 `stt.Transcriber` 재사용 대신 `WhisperModel` 직접 호출(STTConfig 파라미터
  미러링) — bias arm 인자(initial_prompt/hotwords) 때문. 런타임 코드는 불변.
- 숫자 표기 통일(10↔십) 미구현 — CER 페널티로 수용.
- manifest 는 스펙의 snr10/snr5/snr0 라벨 대신 `condition + snr_db` 인코딩 사용.

## 알려진 한계

- ReSpeaker ch0·실차 캐빈 음향 미반영(실차 녹음 가능 시 v1).
- TTS 절대 CER 은 편향 가능 — 긴급문구 지표는 arm 간 상대 비교용.
- 확성기는 근사 시뮬(밴드패스+클립+반사 1개), 실측 IR 아님.
- 숫자 표기(10↔십)는 CER 페널티로 남음.
- 발화 단위 파일 평가라 VAD/버퍼링 미경유(배포 경로와 차이).

## baseline 측정 결과 (Task 10 에서 기입)

| arm | CER(전체) | corpus clean | corpus noise@5 | corpus noise@0 | 키워드 히트율 | 비고 |
|---|---|---|---|---|---|---|
| small | | | | | | |
| small-bias | | | | | | |
| medium | | | | | | |
| medium-bias | | | | | | |
| large-v3-turbo | | | | | | (Jetson 또는 GPU 머신) |

게이트 판단은 corpus(실음성) 컬럼 기준 — TTS 절대치는 상대 비교용.

## Go/No-Go 판단 (측정 후 기입)

기준(스펙 §4): SNR≤5dB CER 이 clean 대비 2배↑ 열화 or 키워드 히트율 <90%.
hotwords arm 만으로 충족되면 파인튜닝 보류.

- 판단: (기입)
- 근거: (기입)
