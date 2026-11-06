"""
classify.py — 소리 분류만 (접근/방향 제외).

분류 모듈(classifier.infer)만 돌려 {siren / horn / normal_traffic} + 신뢰도를 출력한다.
도플러·방향을 안 돌리므로 가볍고, "이 소리가 뭔지"만 빠르게 본다.

사용:
  python classify.py --demo                       # 합성음
  python classify.py --wav 소리.wav                # 파일
  python classify.py --mic                         # 기본 마이크 (1채널)
  python classify.py --mic --device N              # 특정 마이크
  python classify.py --mic --device N --channels 6 # ReSpeaker(6채널) → ch0(처리채널) 사용

ReSpeaker 메모: USB로 6채널이 나오고 분류에는 ch0(빔포밍 처리채널)만 쓴다.
               → --channels 6 으로 열고 첫 채널만 분류기에 넣는다.
"""

from __future__ import annotations

import argparse

from core.types import AudioChunk, SAMPLE_RATE
from audio.capture import iter_chunks_from_array, load_wav
from classifier import infer


def _show(i: int, r) -> None:
    tag = "긴급" if r.is_emergency else "-"
    ko = {"siren": "사이렌", "horn": "경적", "normal_traffic": "일반 소음"}[r.label.value]
    sub = ""
    if r.subtype is not None:               # 사이렌이면 차종 (구급/경찰/소방/긴급차량)
        ko_sub = {"ambulance": "구급차", "police": "경찰차",
                  "fire": "소방차", "unknown": "긴급차량"}[r.subtype.value]
        sub = f" · {ko_sub}({r.subtype_confidence:.2f})"
    print(f"[{i:4d}] {ko:8s} ({r.label.value:14s}) conf={r.confidence:.2f}  {tag}{sub}")


def mic_chunks(device, channels, sr=SAMPLE_RATE, chunk_s=1.0):
    """마이크에서 1초씩 읽어 ch0만 분류기에 전달."""
    import sounddevice as sd
    n = int(sr * chunk_s)
    with sd.InputStream(samplerate=sr, channels=channels, dtype="float32", device=device) as stream:
        while True:
            data, _ = stream.read(n)
            yield AudioChunk(samples=data[:, 0].copy(), sample_rate=sr)  # ch0 = 분류용


def run(chunks) -> None:
    for i, ch in enumerate(chunks):
        _show(i, infer(ch))


def main() -> None:
    ap = argparse.ArgumentParser(description="소리 분류 전용")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--demo", action="store_true", help="합성음 분류")
    g.add_argument("--wav", type=str, help="WAV 파일 분류")
    g.add_argument("--mic", action="store_true", help="실시간 마이크 분류")
    ap.add_argument("--device", type=int, default=None, help="마이크 장치 인덱스")
    ap.add_argument("--channels", type=int, default=1, help="열 채널 수 (ReSpeaker=6)")
    args = ap.parse_args()

    print("※ 분류 모델은 5초 윈도우 → 시작 ~5초는 버퍼 채우는 중이라 불안정")
    if args.demo:
        from main import _synth_passby
        run(iter_chunks_from_array(_synth_passby()))
    elif args.wav:
        run(iter_chunks_from_array(load_wav(args.wav)))
    else:
        print(f"== MIC 분류 (device={args.device}, channels={args.channels}, Ctrl+C 종료) ==")
        run(mic_chunks(args.device, args.channels))


if __name__ == "__main__":
    main()
