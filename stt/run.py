"""
STT 데모 러너 (노트북 단계 검증용).

사용법:
    python -m stt.run --wav some_speech.wav        # 파일 한 방 인식
    python -m stt.run --mic                         # 노트북 마이크 실시간(발화 단위)
    python -m stt.run --mic --respeaker             # Jetson + ReSpeaker(ch0) 실시간
    python -m stt.run --wav file.wav --model small  # 모델 크기 지정
    python -m stt.run --mic --cpu                   # GPU 무시하고 CPU 강제

실시간 모드는 라이브 음량 미터 + 상태(·대기 / 🎙️듣는 중 / ⏳변환 중)를 보여 준다.
끄려면 --no-meter.

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
from .transcriber import Transcriber, transcribe_array, _rms, _to_mono


def _meter(rms: float, threshold: float, width: int = 16) -> str:
    """입력 음량 막대 + VAD 임계 통과 표시."""
    filled = max(0, min(width, int((rms / 0.3) * width)))   # 0~0.3 RMS 를 막대로
    bar = "█" * filled + "·" * (width - filled)
    mark = "🎙️" if rms >= threshold else "  "               # 임계 넘으면(=음성 인정) 마이크 표시
    return f"{mark}|{bar}| {rms:5.3f}"


def _status(line: str) -> None:
    print("\r" + line + "        ", end="", file=sys.stderr, flush=True)


def _clear() -> None:
    print("\r" + " " * 64 + "\r", end="", file=sys.stderr, flush=True)


def run_wav(path: str, cfg: STTConfig) -> None:
    print(f"엔진: {cfg.engine}({cfg.model_size}) | 파일: {path}")
    samples = capture.load_wav(path)
    print("⏳ 변환 중…", file=sys.stderr, flush=True)
    result = transcribe_array(samples, sample_rate=cfg.sample_rate, config=cfg)
    print(result.to_korean())


def run_mic(cfg: STTConfig, respeaker: bool = False, device: int | None = None,
            meter: bool = True) -> None:
    src = "ReSpeaker(ch0)" if respeaker else "기본 마이크"
    print(f"엔진: {cfg.engine}({cfg.model_size}) | {src} 입력 시작 (Ctrl+C 종료)\n")

    # _flush 직전 'transcribing' 콜백 → 변환 중 표시. 같은 transcribe() 호출에서
    # 콜백이 먼저 뜨고 엔진(블로킹) 후 결과가 돌아온다.
    flushing = {"on": False}

    def on_status(s: str) -> None:
        if s == "transcribing":
            flushing["on"] = True
            _status("⏳ 변환 중…")

    transcriber = Transcriber(config=cfg, on_status=on_status)
    stream = (capture.iter_chunks_from_respeaker(device=device) if respeaker
              else capture.iter_chunks_from_mic(device=device))
    try:
        for chunk in stream:
            flushing["on"] = False
            rms = _rms(_to_mono(chunk.samples))
            result = transcriber.transcribe(chunk)

            if flushing["on"]:                       # 이번 청크에서 실제 변환이 일어남
                _clear()
                if result.text:
                    print(result.to_korean())
                else:
                    print("(인식 결과 없음 — 더 또렷이/가까이 말해보세요)")
            elif meter:                              # 변환 안 했으면 라이브 미터
                if result.is_speech:
                    _status("🎙️ 듣는 중  " + _meter(rms, cfg.vad_rms_threshold))
                else:
                    _status("·  대기     " + _meter(rms, cfg.vad_rms_threshold))
    except KeyboardInterrupt:
        _clear()
        tail = transcriber.flush()                   # 종료 직전 남은 발화 처리
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
    ap.add_argument("--accuracy", action="store_true", help="정확도 우선(beam↑, 속도 양보)")
    ap.add_argument("--beam", type=int, default=None, help="beam_size 직접 지정(정확도↑/속도↓)")
    ap.add_argument("--vad", type=float, default=None, help="VAD 임계값(기본 0.02). 조용한 실내면 ↓, 도로면 ↑")
    ap.add_argument("--normalize", action="store_true", help="RMS 정규화 켜기(입력이 일관되게 작을 때만)")
    ap.add_argument("--no-meter", action="store_true", help="라이브 음량 미터/상태 표시 끄기")
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
    if args.accuracy:                  # 정확도 우선 프로파일
        cfg = STTConfig.for_accuracy(cfg)
    if args.beam is not None:
        cfg.beam_size = args.beam
    if args.vad is not None:
        cfg.vad_rms_threshold = args.vad
    if args.normalize:
        cfg.normalize_audio = True

    if args.wav:
        run_wav(args.wav, cfg)
    else:
        run_mic(cfg, respeaker=args.respeaker, device=args.device, meter=not args.no_meter)


if __name__ == "__main__":
    main()
