"""
통합 파이프라인 (MVP 데모).

세 모듈을 합쳐 최종 결과를 출력한다:
  ① 분류(classifier, 실제) + ② 방향(doa, 스텁) + ③ 접근(approach, 스텁)
  → FusedResult → "사이렌, 방향 미상, 이동 미상"

사용법:
    python -m pipeline.run --wav data/UrbanSound8K/audio/fold10/xxxx.wav
    python -m pipeline.run --mic          # 노트북 마이크 실시간
    python -m pipeline.run --wav <file> --exp yamnet_frozen_mlp   # 모델 지정
"""

from __future__ import annotations

import argparse
import sys

try:  # Windows 콘솔(cp949)에서도 한글 출력이 깨지거나 죽지 않게
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from core.types import FusedResult
from audio import capture
from classifier.infer import Classifier
from doa.estimator import estimate_direction
from approach.detector import ApproachDetector


def _fuse(clf, approach, chunk) -> FusedResult:
    return FusedResult(
        sound=clf.classify(chunk),            # ① 실제 분류
        direction=estimate_direction(chunk),  # ② 방향 (스텁 → unknown)
        approach=approach.update(chunk),      # ③ 접근 (스텁 → unknown)
    )


def run_wav(path: str, exp: str | None = None):
    clf = Classifier(exp)
    approach = ApproachDetector()
    print(f"모델: {clf.exp.name} | 파일: {path}\n")
    samples = capture.load_wav(path)
    for i, chunk in enumerate(capture.iter_chunks_from_array(samples), start=1):
        fused = _fuse(clf, approach, chunk)
        print(f"[{i}초] {fused.to_korean()}  (신뢰도 {fused.sound.confidence:.2f})")


def run_mic(exp: str | None = None):
    clf = Classifier(exp)
    approach = ApproachDetector()
    print(f"모델: {clf.exp.name} | 마이크 입력 시작 (Ctrl+C 종료)\n")
    try:
        for chunk in capture.iter_chunks_from_mic():
            fused = _fuse(clf, approach, chunk)
            print(f"{fused.to_korean()}  (신뢰도 {fused.sound.confidence:.2f})")
    except KeyboardInterrupt:
        print("\n종료.")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--wav", type=str, help="분석할 WAV 파일")
    g.add_argument("--mic", action="store_true", help="마이크 실시간")
    ap.add_argument("--exp", type=str, default=None, help="사용할 모델(미지정 시 best)")
    args = ap.parse_args()

    if args.wav:
        run_wav(args.wav, args.exp)
    else:
        run_mic(args.exp)


if __name__ == "__main__":
    main()
