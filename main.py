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
import sys
import warnings

import numpy as np

# 무음·극단 입력 시 멜 스펙트로그램 matmul 에서 뜨는 무해한 RuntimeWarning 억제.
# (우리 classifier 와 faster-whisper 양쪽 — 분류·인식 결과엔 영향 없음)
warnings.filterwarnings("ignore", category=RuntimeWarning, message=r".*matmul.*")

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


def run_stream(chunks, stt_worker=None, hud=None) -> None:
    pipe = Pipeline(stt_worker=stt_worker)
    try:
        for i, chunk in enumerate(chunks):
            fused = pipe.process(chunk)
            print(f"[{i:3d}] {fused.to_korean():<22s}  (분류 conf={fused.sound.confidence:.2f})")
            if hud is not None:
                hud.update(fused)
                if hud.stopped:          # HUD 창이 닫히면 파이프라인도 종료
                    break
    finally:
        if stt_worker is not None:
            stt_worker.stop()


def _run_with_hud(source, stt_worker, args) -> None:
    """--hud: 파이프라인을 백그라운드 스레드로, pygame 렌더 루프를 메인 스레드에서."""
    try:
        from hud.config import HudConfig
        from hud.display import HudDisplay
    except ImportError as e:
        print(f"[hud] pygame 미설치 — HUD 없이 콘솔로 계속: {e}\n"
              "  설치: pip install pygame", file=sys.stderr)
        run_stream(source, stt_worker)
        return

    import threading
    cfg = HudConfig(fullscreen=not args.hud_windowed, reflect=args.hud_flip)
    hud = HudDisplay(cfg)
    # --demo/--wav처럼 유한 소스면 파이프라인 스레드는 소스 소진 시 끝나지만, HUD 창은
    # 의도적으로 ESC/Q 입력 전까지 마지막 프레임을 유지한다 (daemon 스레드라 종료를 막지 않음).
    worker = threading.Thread(
        target=run_stream, args=(source, stt_worker, hud),
        name="pipeline", daemon=True)
    worker.start()
    try:
        hud.run()                # 메인 스레드 블로킹 (종료 시 반환)
    finally:
        hud.stop()               # 파이프라인 루프도 멈추라고 신호
        worker.join(timeout=2.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="EmergencySoundAssist 파이프라인")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--demo", action="store_true", help="합성 통과 시연")
    g.add_argument("--wav", type=str, help="WAV 파일 경로")
    g.add_argument("--mic", action="store_true", help="실시간 마이크")
    ap.add_argument("--device", type=int, default=None,
                    help="마이크 장치 인덱스 (미지정 시: 다채널이면 ReSpeaker 자동 탐지 → 기본 장치)")
    ap.add_argument("--channels", type=int, default=1,
                    help="마이크 채널 수 (ReSpeaker=6 → ch0 분류·접근, ch1~4 방향)")
    ap.add_argument("--stt", action="store_true",
                    help="평상시 음성→자막(STT). 사이렌·경적일 땐 자동 멈춤. "
                         "faster-whisper 필요(pip install -r stt/requirements.txt)")
    ap.add_argument("--stt-model", default=None,
                    help="STT 모델 크기/경로 (tiny/base/small/medium/…). 미지정 시 STTConfig 기본(medium). "
                         "GPU 없는 보드·노트북은 small 이하 권장 — medium 은 CPU 실시간 불가")
    ap.add_argument("--hud", action="store_true",
                    help="HUD 화면 출력(pygame). 전체화면 기본. 긴급이면 방향 레이더, 평상시엔 자막.")
    ap.add_argument("--hud-windowed", action="store_true",
                    help="HUD를 창 모드로(노트북 개발용). --hud 와 함께.")
    ap.add_argument("--hud-flip", action="store_true",
                    help="HUD 반사(윈드실드) 모드 — 상하반전 시작. 런타임 F키로 토글.")
    args = ap.parse_args()

    stt_worker = None
    if args.stt:
        from stt.transcriber import Transcriber
        from stt.config import STTConfig
        from stt.worker import STTWorker
        cfg = STTConfig()
        if args.stt_model:
            cfg.model_size = args.stt_model
        stt_worker = STTWorker(Transcriber(config=cfg))
        stt_worker.start()
        print(f"[stt] 평상시 자막 ON (모델 {cfg.model_size}, 백그라운드 스레드) — 사이렌·경적일 땐 자동 멈춤")

    if args.demo:
        print("== DEMO: 합성 사이렌 통과 (60km/h, 측면 8m) ==")
        source = iter_chunks_from_array(_synth_passby())
    elif args.wav:
        print(f"== WAV: {args.wav} ==")
        source = iter_chunks_from_array(load_wav(args.wav))
    else:
        print("== MIC: 실시간 (Ctrl+C 종료) ==")
        source = iter_chunks_from_mic(device=args.device, channels=args.channels)

    if args.hud:
        _run_with_hud(source, stt_worker, args)
    else:
        run_stream(source, stt_worker)


if __name__ == "__main__":
    main()
