"""
STT 데모 러너 (노트북 단계 검증용).

사용법:
    python -m stt.run --wav some_speech.wav        # 파일 한 방 인식
    python -m stt.run --mic                         # 노트북 마이크 실시간(발화 단위)
    python -m stt.run --mic --respeaker             # Jetson + ReSpeaker(ch0) 실시간
    python -m stt.run --wav file.wav --model small  # 모델 크기 지정
    python -m stt.run --mic --cpu                   # GPU 무시하고 CPU 강제

faster-whisper 설치 필요:  pip install -r stt/requirements.txt
Jetson 배포 절차:          docs/stt/jetson.md
"""

from __future__ import annotations

import argparse
import sys

try:  # Windows 콘솔(cp949)에서도 한글이 깨지지 않게
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from audio import capture
from .config import STTConfig
from .transcriber import Transcriber, transcribe_array


def run_wav(path: str, cfg: STTConfig) -> None:
    print(f"엔진: {cfg.engine}({cfg.model_size}) | 파일: {path}\n")
    samples = capture.load_wav(path)
    result = transcribe_array(samples, sample_rate=cfg.sample_rate, config=cfg)
    print(result.to_korean())
    if result.keywords:
        print(f"  └ 긴급 키워드: {', '.join(result.keywords)}")


def run_mic(cfg: STTConfig, respeaker: bool = False, device: int | None = None) -> None:
    src = "ReSpeaker(ch0)" if respeaker else "기본 마이크"
    print(f"엔진: {cfg.engine}({cfg.model_size}) | {src} 입력 시작 (Ctrl+C 종료)\n")
    transcriber = Transcriber(config=cfg)
    stream = (capture.iter_chunks_from_respeaker(device=device) if respeaker
              else capture.iter_chunks_from_mic(device=device))
    try:
        for chunk in stream:
            result = transcriber.transcribe(chunk)
            if result.is_speech and result.text:   # 발화가 완성된 순간만 출력
                print(result.to_korean())
    except KeyboardInterrupt:
        tail = transcriber.flush()                 # 종료 직전 남은 발화 처리
        if tail.is_speech and tail.text:
            print(tail.to_korean())
        print("\n종료.")


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--wav", type=str, help="인식할 WAV 파일")
    g.add_argument("--mic", action="store_true", help="마이크 실시간")
    ap.add_argument("--model", type=str, default=None, help="모델 크기(tiny/base/small...)")
    ap.add_argument("--lang", type=str, default=None, help="언어 코드(예: ko). 미지정 시 설정값")
    ap.add_argument("--respeaker", action="store_true", help="ReSpeaker 6채널 ch0 입력(Jetson)")
    ap.add_argument("--device", type=int, default=None, help="입력 장치 인덱스(미지정 시 자동)")
    ap.add_argument("--cpu", action="store_true", help="GPU 무시하고 CPU 강제")
    ap.add_argument("--threads", type=int, default=None, help="CPU 스레드 수(Orin 6코어면 6). 속도↑")
    ap.add_argument("--accuracy", action="store_true", help="정확도·범위 우선(beam↑+프롬프트, 속도 양보)")
    ap.add_argument("--beam", type=int, default=None, help="beam_size 직접 지정(정확도↑/속도↓)")
    args = ap.parse_args()

    cfg = STTConfig()
    if args.model:
        cfg.model_size = args.model
    if args.lang:
        cfg.language = args.lang
    if args.cpu:                       # GPU 가 있어도 CPU 로 강제
        cfg.device, cfg.compute_type = "cpu", "int8"
    if args.threads is not None:
        cfg.cpu_threads = args.threads
    if args.accuracy:                  # 정확도/범위 우선 프로파일
        cfg = STTConfig.for_accuracy(cfg)
    if args.beam is not None:
        cfg.beam_size = args.beam

    if args.wav:
        run_wav(args.wav, cfg)
    else:
        run_mic(cfg, respeaker=args.respeaker, device=args.device)


if __name__ == "__main__":
    main()
