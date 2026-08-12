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


def _stt_diag(worker, state: dict) -> None:
    """STT 워커의 지연·오류·사망을 콘솔(stderr)에만 알린다. HUD 에는 올리지 않는다.

    젯슨처럼 느린 보드에서 STT 가 밀리면 자막만 조용히 안 나온다. 그게 모델이 느린
    건지 워커가 죽은 건지 화면만 봐선 구분이 안 되므로 상태 변화를 한 번씩 찍는다.
    state 는 호출 간 마지막 값 보관용 — 같은 상태를 매 틱 반복 출력하지 않는다.
    """
    if worker is None or not hasattr(worker, "status"):
        return
    st = worker.status()
    error = st.get("last_error")
    if error and error != state["error"]:
        print(f"[stt] 오류(워커는 유지): {error}", file=sys.stderr)
    state["error"] = error
    dropped = int(st.get("dropped_chunks", 0))
    if dropped > state["dropped"]:
        print(f"[stt] 지연: 입력 {dropped - state['dropped']}개 생략", file=sys.stderr)
    state["dropped"] = dropped
    if not st.get("alive", True) and not state["dead"]:
        print("[stt] 워커 스레드 종료됨 — 이후 자막 없음", file=sys.stderr)
        state["dead"] = True


def run_stream(chunks, stt_worker=None, hud=None, sender=None, collector=None) -> None:
    pipe = Pipeline(stt_worker=stt_worker)
    stt_state = {"error": None, "dropped": 0, "dead": False}
    try:
        for i, chunk in enumerate(chunks):
            fused = pipe.process(chunk)
            print(f"[{i:3d}] {fused.to_korean():<22s}  (분류 conf={fused.sound.confidence:.2f})")
            _stt_diag(stt_worker, stt_state)
            # 수집·BLE 는 1초 틱에서만 — 청크가 0.25초라 매 청크로 보내면 det_flags 가
            # 0.25초 틱이 되어 하류 도구의 '1틱=1초' 규약이 깨지고, BLE 는 4배로 쏜다.
            # 오디오는 손실 없이 모아서 넘어간다(Pipeline 이 누적해 tick_chunk 로 준다).
            if collector is not None and pipe.full_tick:
                collector.on_result(pipe.tick_chunk, pipe.tick_raw)
            if sender is not None and pipe.full_tick:
                sender.send_fused(fused)
            if hud is not None:
                hud.update(fused)
                if hud.stopped:          # HUD 창이 닫히면 파이프라인도 종료
                    break
    finally:
        if collector is not None:        # 열려 있던 클립을 저장하고 세션 요약을 남긴다
            msg = collector.close()      # 멱등 — 메인 스레드가 이미 닫았으면 None
            if msg:
                print(msg)
        if sender is not None:
            sender.close()
        if stt_worker is not None:
            stt_worker.stop()


def _run_with_hud(source, stt_worker, args, sender=None, doa_poller=None, collector=None) -> None:
    """--hud: 파이프라인을 백그라운드 스레드로, pygame 렌더 루프를 메인 스레드에서."""
    try:
        from hud.config import HudConfig
        from hud.display import HudDisplay
    except ImportError as e:
        print(f"[hud] pygame 미설치 — HUD 없이 콘솔로 계속: {e}\n"
              "  설치: pip install pygame", file=sys.stderr)
        run_stream(source, stt_worker, sender=sender, collector=collector)
        return

    import threading
    cfg = HudConfig(fullscreen=not args.hud_windowed, reflect=args.hud_flip)
    hud = HudDisplay(cfg)
    if doa_poller:
        hud.set_doa_poller(doa_poller)
    if collector is not None:
        hud.set_collector(collector)

    # --demo/--wav처럼 유한 소스면 파이프라인 스레드는 소스 소진 시 끝나지만, HUD 창은
    # 의도적으로 ESC/Q 입력 전까지 마지막 프레임을 유지한다 (daemon 스레드라 종료를 막지 않음).
    worker = threading.Thread(
        target=run_stream, args=(source, stt_worker, hud, sender, collector),
        name="pipeline", daemon=True)
    worker.start()
    try:
        hud.run()                # 메인 스레드 블로킹 (종료 시 반환)
    finally:
        hud.stop()               # 파이프라인 루프도 멈추라고 신호
        worker.join(timeout=2.0)
        # 파이프라인은 데몬이라 join 타임아웃 뒤 인터프리터 종료가 저장 도중
        # 스레드를 동결시킬 수 있다. 수집 데이터는 메인 스레드에서 닫음을 보증한다
        # (멱등 — 파이프라인 finally 가 이미 닫았으면 None).
        if collector is not None:
            msg = collector.close()
            if msg:
                print(msg)


def main() -> None:
    ap = argparse.ArgumentParser(description="EmergencySoundAssist 파이프라인")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--demo", action="store_true", help="합성 통과 시연")
    g.add_argument("--wav", type=str, help="WAV 파일 경로")
    g.add_argument("--mic", action="store_true", help="실시간 마이크")
    ap.add_argument("--device", type=int, default=None,
                    help="마이크 장치 인덱스 (미지정 시: 다채널이면 ReSpeaker 자동 탐지 → 기본 장치)")
    ap.add_argument("--channels", type=int, default=None,
                    help="마이크 채널 수 (ReSpeaker=6 → ch0 분류·접근, ch1~4 방향). "
                         "미지정 시 자동: ReSpeaker 가 보이면 6, 아니면 1 — 젯슨에서 "
                         "플래그 없이 틀어도 방향/수집이 제대로 돌게.")
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
    ap.add_argument("--ble", action="store_true",
                    help="감지 결과를 BLE로 폰(→워치 미러링 진동)에 전송 (bleak 필요). --hud 와 함께 쓸 수 있다.")
    ap.add_argument("--watch-mac", default=None,
                    help="BLE 대상 MAC 직접 지정(미지정 시 서비스 UUID로 폰 자동 검색)")
    ap.add_argument("--collect", action="store_true",
                    help="사이렌 학습 데이터 수집 모드: 감지되면 자동 녹음(프리롤 포함), "
                         "HUD 차종 버튼(1/2/3/u·터치)으로 라벨. 미감지 때 버튼을 누르면 "
                         "수동 녹음(미검출 표본). --hud 와 함께 권장.")
    ap.add_argument("--place", default="미지정",
                    help="--collect 세션 폴더명에 들어갈 장소. 예: 소방서앞")
    ap.add_argument("--collect-buttons", action="store_true",
                    help="수집 라벨 버튼 띠를 화면에 표시(개발용). 기본은 조용한 표시 — "
                         "제품 화면 그대로 두고 키보드로만 라벨을 받는다.")
    ap.add_argument("--collect-out", default="data/collect_sessions",
                    help="--collect 세션 폴더를 만들 위치")
    args = ap.parse_args()

    # 채널 자동 결정 — 젯슨(ReSpeaker)에서 플래그 없이 틀어도 6채널로 열리게.
    # 명시(--channels N)가 항상 이긴다. 마이크가 아니면 채널은 쓰이지 않는다.
    if args.channels is None:
        args.channels = 1
        if args.mic:
            from audio.capture import _find_respeaker_index
            try:
                if _find_respeaker_index() is not None:
                    args.channels = 6
            except Exception:
                pass                     # sounddevice 문제는 캡처 시작에서 제대로 드러난다
            print(f"[audio] 채널 자동 선택: {args.channels}"
                  + (" (ReSpeaker 감지)" if args.channels == 6 else " (ReSpeaker 미감지)"))

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

    # --ble 일 때만 BLE 송신기 시작 (없으면 sender=None → 전송 생략, 지연 import)
    sender = None
    if args.ble:
        try:
            import bleak  # noqa: F401 — 시작 시 의존성 확인
        except ImportError as exc:
            # 확인하지 않으면 BLE 스레드가 뒤늦게 트레이스백을 뱉고 조용히 죽는다.
            raise SystemExit(
                "[ble] bleak이 없습니다. "
                "먼저 `.venv/bin/python -m pip install bleak`을 실행하세요."
            ) from exc
        from notify import BleSender
        sender = BleSender(address=args.watch_mac)
        sender.start()
        print("== BLE: 전송 활성화 (폰 GATT 서버를 서비스 UUID로 검색) ==")

    collector = None
    if args.collect:
        from collect import SirenCollector
        collector = SirenCollector(out_root=args.collect_out, place=args.place,
                                   show_buttons=args.collect_buttons)
        print(f"[collect] 세션: {collector.session}")
        print("[collect] 감지되면 자동 녹음 · 키 1 구급차 / 2 경찰차 / 3 소방차 / u 차종모름"
              " / h 경적 / n 사이렌아님 / z 취소 · 미감지 때 키 = 수동 녹음")
        print("[collect] 화면은 제품 그대로 — 라벨은 키보드로만 받는다"
              f"{' (버튼 띠 표시 중)' if args.collect_buttons else ''}")
        if not args.hud:
            print("[collect] 경고: --hud 없이는 라벨 버튼이 없다 — 자동 클립만 "
                  "unlabeled 로 쌓인다. --hud 를 함께 켤 것.", file=sys.stderr)

    doa_poller = None
    if args.hud:
        from doa.doa_poller import DoaPoller
        doa_poller = DoaPoller()
        doa_poller.start()

    try:
        if args.demo:
            print("== DEMO: 합성 사이렌 통과 (60km/h, 측면 8m) ==")
            source = iter_chunks_from_array(_synth_passby(), doa_callback=doa_poller.push if doa_poller else None)
        elif args.wav:
            print(f"== WAV: {args.wav} ==")
            source = iter_chunks_from_array(load_wav(args.wav), doa_callback=doa_poller.push if doa_poller else None)
        else:
            print("== MIC: 실시간 (Ctrl+C 종료) ==")
            source = iter_chunks_from_mic(device=args.device, channels=args.channels, doa_callback=doa_poller.push if doa_poller else None)
            if collector is not None:
                # 수집 중엔 클립 저장(wav 쓰기)이 파이프라인 스레드를 수백 ms 붙잡을
                # 수 있다. 캡처를 별도 스레드+큐로 떼어 그 순간의 입력 유실(다음
                # 사이렌의 프리롤이 끊기는 것)을 막는다.
                from audio.capture import iter_chunks_threaded
                source = iter_chunks_threaded(source)
    
        if args.hud:
            _run_with_hud(source, stt_worker, args, sender, doa_poller, collector)
        else:
            run_stream(source, stt_worker, sender=sender, collector=collector)
    finally:
        if doa_poller:
            doa_poller.stop()


if __name__ == "__main__":
    main()
