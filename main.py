"""
EmergencySoundAssist 실시간 메인 루프.

사용:
  python main.py --demo            # 합성 사이렌 통과로 전체 파이프라인 시연 (마이크/파일 불필요)
  python main.py --wav PATH        # WAV 파일을 청크로 흘려 처리
  python main.py --mic             # 실시간 마이크 (sounddevice 필요)

현재 상태 (모듈 채움 정도):
  ① classifier : placeholder 휴리스틱 (학습 모델 미연결)
  ② doa        : stub (방향 미상)
  ③ approach   : 실시간 도플러 구현 완료 ✔
파이프라인·main은 이 골격으로 동작하며, ①②를 진짜 구현으로 교체해 나가면 된다.
"""

from __future__ import annotations

import argparse

import numpy as np

from core.types import AudioChunk, SAMPLE_RATE
from audio.capture import iter_chunks_from_array, iter_chunks_from_mic, load_wav
from pipeline import Pipeline

_C = 343.0  # 음속 m/s


def _synth_passby(sr=SAMPLE_RATE, f0=700.0, v_kmh=60.0, d=8.0, dur=10.0, snr_db=20.0):
    """데모용: 정지 톤 f0를 속도 v·측면거리 d 통과 신호로 합성 (도플러 + 1/r)."""
    v = v_kmh / 3.6
    t = np.arange(0.0, dur, 1.0 / sr)
    x = v * (t - dur / 2.0)
    r = np.sqrt(d ** 2 + x ** 2)
    v_closing = -(v * x / r)
    f_obs = f0 * _C / (_C - v_closing)
    amp = d / r
    sig = amp * np.sin(np.cumsum(2 * np.pi * f_obs / sr))
    rng = np.random.default_rng(0)
    n = rng.standard_normal(sig.size)
    sp = np.mean(sig ** 2) + 1e-12
    n *= np.sqrt(sp / (10 ** (snr_db / 10)) / (np.mean(n ** 2) + 1e-12))
    return (sig + n).astype(np.float32)


def run_stream(chunks) -> None:
    pipe = Pipeline()
    for i, chunk in enumerate(chunks):
        fused = pipe.process(chunk)
        print(f"[{i:3d}] {fused.to_korean():<22s}  (분류 conf={fused.sound.confidence:.2f})")


def main() -> None:
    ap = argparse.ArgumentParser(description="EmergencySoundAssist 파이프라인")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--demo", action="store_true", help="합성 통과 시연")
    g.add_argument("--wav", type=str, help="WAV 파일 경로")
    g.add_argument("--mic", action="store_true", help="실시간 마이크")
    ap.add_argument("--device", type=int, default=None, help="마이크 장치 인덱스")
    ap.add_argument("--channels", type=int, default=1,
                    help="마이크 채널 수 (ReSpeaker=6 → ch0 분류·접근, ch1~4 방향)")
    args = ap.parse_args()

    if args.demo:
        print("== DEMO: 합성 사이렌 통과 (60km/h, 측면 8m) ==")
        sig = _synth_passby()
        run_stream(iter_chunks_from_array(sig))
    elif args.wav:
        print(f"== WAV: {args.wav} ==")
        sig = load_wav(args.wav)
        run_stream(iter_chunks_from_array(sig))
    else:
        print("== MIC: 실시간 (Ctrl+C 종료) ==")
        run_stream(iter_chunks_from_mic(device=args.device, channels=args.channels))


if __name__ == "__main__":
    main()
