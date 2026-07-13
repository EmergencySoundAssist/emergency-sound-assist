"""
EmergencySoundAssist 실시간 메인 루프.

사용:
  python main.py --demo            # 합성 사이렌 통과로 전체 파이프라인 시연 (마이크/파일 불필요)
  python main.py --wav PATH        # WAV 파일을 청크로 흘려 처리
  python main.py --mic             # 실시간 마이크 (sounddevice 필요)

하이브리드 런타임 (exp/hybrid-runtime — 석우 경보엔진 + 우리 방향/STT):
  검출  : 이중 창 (5s 확정 + 2s 예비 87f) · 로짓 마진 → alert 상태기계 (ONSET/REMIND/CLEAR)
  차종  : yt 실채널 파인튜닝판 + 6초 다수결 (사이렌 ON 중)
  움직임: 음량 기울기 (approach.detector — 접근/멀어짐/유지 + 빠르기) — 우리 것
  방향  : SRP-PHAT (ch1~4, 긴급 중) — 우리 것
  STT   : webrtcvad→Whisper 백그라운드 — 우리 것 (★1초 뭉치 feed)
  tick  : 0.15s 기본 (--tick)
"""

from __future__ import annotations

import argparse
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


def run_stream(chunks, stt_worker=None, dt: float = 0.15, view: str = "console") -> None:
    """하이브리드 루프. view="console": 상세 로그(영구 줄+상태줄). view="dashboard": 제자리 갱신 패널."""
    from pipeline import alert

    pipe = Pipeline(stt_worker=stt_worker, dt=dt)

    if view == "dashboard":                          # 제자리 갱신 대시보드 (보기 쉬움)
        dash = alert.DashboardSink()
        try:
            for chunk in chunks:
                ev, info = pipe.process(chunk)
                dash.update(ev, info)
        finally:
            dash.close()
            if stt_worker is not None:
                stt_worker.stop()
        return

    sink = alert.make_sink("console")
    shown_dir = False                               # 경보당 방향 1회 표시 (튀는 도배 방지)
    try:
        for chunk in chunks:
            ev, info = pipe.process(chunk)
            sink.emit(ev)                           # ONSET/REMIND/CLEAR 만 영구 출력
            if ev.clear:
                shown_dir = False
            if info.get("direction") is not None and (ev.onset or not shown_dir):
                ang = f"({info['angle']:.0f}°)" if info.get("angle") is not None else ""
                print(f"\r\033[K         ↳ 방향 {info['direction'].value} {ang}", flush=True)
                shown_dir = True
            sp = info.get("speech")
            if sp is not None and sp.is_speech and sp.text:
                print(f"\r\033[K         ↳ 자막: \"{sp.text}\"", flush=True)
            if info:
                sink.tick(info["m_siren"], info["state"], info["level"],
                          info.get("risk"), info.get("dir_raw"), info.get("gauge"))
    finally:
        sink.close()
        if stt_worker is not None:
            stt_worker.stop()


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
    ap.add_argument("--tick", type=float, default=0.15,
                    help="tick 간격(초). 기본 0.15 (석우 런타임과 동일 — 예비경보 ~1.8s)")
    ap.add_argument("--view", choices=["console", "dashboard"], default="console",
                    help="출력 형식: console(상세 로그) / dashboard(제자리 갱신 패널, 보기 쉬움)")
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
        print(f"== DEMO: 합성 사이렌 통과 (60km/h, 측면 8m) · tick {args.tick}s ==")
        sig = _synth_passby()
        run_stream(iter_chunks_from_array(sig, chunk_seconds=args.tick), stt_worker, args.tick, args.view)
    elif args.wav:
        print(f"== WAV: {args.wav} · tick {args.tick}s ==")
        sig = load_wav(args.wav)
        run_stream(iter_chunks_from_array(sig, chunk_seconds=args.tick), stt_worker, args.tick, args.view)
    else:
        print(f"== MIC: 실시간 · tick {args.tick}s (Ctrl+C 종료) ==")
        run_stream(iter_chunks_from_mic(device=args.device, channels=args.channels,
                                        chunk_seconds=args.tick), stt_worker, args.tick, args.view)


if __name__ == "__main__":
    main()
