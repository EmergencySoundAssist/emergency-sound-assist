# finetune — STT 파인튜닝 준비 (평가셋 v0 + baseline 측정)

담당: 천자민. 설계: [스펙](../docs/superpowers/specs/2026-07-06-stt-finetune-data-eval-design.md),
결과 기록: [docs/stt/finetune.md](../docs/stt/finetune.md)

**목적:** 파인튜닝 전에 "현행 faster-whisper 가 뭘 틀리는지" 먼저 측정한다.
안 틀리면 파인튜닝 안 한다. (go/no-go 게이트는 docs/stt/finetune.md 참고)

## 설치 (개발 머신 전용 — Jetson 금지)

    pip install -r finetune/requirements.txt

## 파이프라인 (순서대로, repo 루트에서)

    python3 -m finetune.noise        # 도로소음 다운로드 (~390MB, 1회)
    python3 -m finetune.tts          # 긴급문구 TTS 합성 (네트워크)
    python3 -m finetune.corpora      # 공개 코퍼스 샘플링 (zeroth+FLEURS)
    python3 -m finetune.build_eval   # 조건 적용 + manifest 생성
    python3 -m finetune.evaluate --model small          # baseline 측정
    python3 -m finetune.evaluate --model small --bias   # hotwords 편향 arm
    python3 -m finetune.evaluate --combine              # arm 비교표 생성

산출물: `data/eval_v0/` (gitignored). 같은 시드(20260706)면 항상 동일하게 재생성된다.

## 사이렌 소음 추가 (선택)

`data/noise/extra/` 에 16kHz mono WAV 를 넣으면 노이즈 풀에 자동 포함된다.
(UrbanSound8K siren 등 — v0 기본 풀은 DEMAND 차량/도로 + MS-SNSD 4종)
