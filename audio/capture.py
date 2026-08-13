"""
공통 오디오 입력 (모든 모듈이 공유).

노트북 개발 단계에서는 노트북 내장 마이크나 WAV 파일로 테스트하고,
나중에 Jetson + ReSpeaker 로 옮길 때 이 파일의 입력 소스만 바꾸면 된다.

의존성: sounddevice(실시간 마이크), soundfile(파일). 둘 다 선택 설치.
파일 재생/테스트만 할 거면 sounddevice 없이도 load_wav() 사용 가능.
"""

from __future__ import annotations

import sys
from typing import Iterator

import numpy as np

from core.types import AudioChunk, SAMPLE_RATE, CHUNK_SECONDS


def load_wav(path: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """WAV 파일을 (n_samples,) float32 모노로 로드. 분류 테스트용."""
    import soundfile as sf  # 지연 import: 파일 안 쓰면 설치 불필요

    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:                    # 다채널이면 평균내서 모노로
        data = data.mean(axis=1)
    if sr != sample_rate:
        data = _resample(data, sr, sample_rate)
    return data


def iter_chunks_from_array(
    samples: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    chunk_seconds: float = CHUNK_SECONDS,
    doa_callback=None,
    doa_interval: float = 0.25,
) -> Iterator[AudioChunk]:
    """긴 오디오 배열을 1초짜리 청크들로 잘라서 내보낸다(파일 테스트용)."""
    import time
    n_total = int(sample_rate * chunk_seconds)
    n_sub = int(sample_rate * doa_interval) if doa_callback else n_total
    
    for start in range(0, len(samples) - n_total + 1, n_total):
        # 배열에서도 실제 시간처럼 시뮬레이션하기 위해 서브청크로 쪼갠다
        cur = start
        while cur < start + n_total:
            end = min(cur + n_sub, start + n_total)
            if doa_callback:
                doa_callback(samples[cur:end])
                time.sleep(doa_interval)  # 파일 시뮬레이션 시 실시간처럼
            cur = end
        
        yield AudioChunk(samples=samples[start:start + n_total], sample_rate=sample_rate)


def iter_chunks_from_mic(
    sample_rate: int = SAMPLE_RATE,
    chunk_seconds: float = CHUNK_SECONDS,
    device: int | None = None,
    channels: int = 1,
    doa_callback=None,
    doa_interval: float = 0.25,
) -> Iterator[AudioChunk]:
    """실시간 마이크에서 1초씩 읽어 청크로 내보낸다.

    노트북: 내장 마이크 1채널 (channels=1, device=None).
    Jetson: ReSpeaker 6채널 (channels=6, device=인덱스) → ch0=분류·접근, ch1~4=방향.

    channels==1 이면 (n,) 모노, 그 이상이면 (n, channels) 다채널을 그대로 담는다.
    (분류·접근은 pipeline 이 ch0 만, 방향은 ch1~4 만 골라 쓴다.)
    device=None 이고 channels>1 이면 ReSpeaker 를 이름으로 자동 탐지한다
    (실패 시 기본 장치 + 경고 라벨).
    """
    import sounddevice as sd  # 지연 import

    device, label = _resolve_input_device(device, channels)
    print(f"[audio] 입력 장치: {label} · {channels}ch {sample_rate}Hz", file=sys.stderr)
    watch = SilenceWatch()
    n_total = int(sample_rate * chunk_seconds)
    n_sub = int(sample_rate * doa_interval) if doa_callback else n_total
    
    with sd.InputStream(samplerate=sample_rate, channels=channels,
                        dtype="float32", device=device) as stream:
        while True:
            full_data = []
            cur_len = 0
            while cur_len < n_total:
                read_n = min(n_sub, n_total - cur_len)
                data, _ = stream.read(read_n)
                if doa_callback:
                    doa_callback(data)
                full_data.append(data)
                cur_len += read_n
            
            data = np.concatenate(full_data, axis=0) if len(full_data) > 1 else full_data[0]
            samples = data[:, 0].copy() if channels == 1 else data.copy()
            warn = watch.update(samples)
            if warn:
                print(warn, file=sys.stderr)
            yield AudioChunk(samples=samples, sample_rate=sample_rate)


def iter_chunks_from_respeaker(
    channel: int = 0,
    num_channels: int = 6,
    sample_rate: int = SAMPLE_RATE,
    chunk_seconds: float = CHUNK_SECONDS,
    device: int | None = None,
) -> Iterator[AudioChunk]:
    """Jetson + ReSpeaker 용 캡처. 6채널로 열어 한 채널만 모노로 내보낸다.

    ReSpeaker XVF-3000 은 USB 로 6채널을 준다(→ docs/hardware.md):
      ch0 = 빔포밍/AEC 처리된 깨끗한 1채널  ← STT 에 가장 적합(기본값)
      ch1~4 = 원본 마이크 / ch5 = 재생 참조
    device=None 이면 'ReSpeaker' 가 이름에 들어간 장치를 자동 탐지한다.
    """
    import sounddevice as sd  # 지연 import

    device, label = _resolve_input_device(device, num_channels)
    print(f"[audio] 입력 장치: {label} · {num_channels}ch→ch{channel} {sample_rate}Hz",
          file=sys.stderr)
    watch = SilenceWatch()
    mic_health.attach(watch)          # main/HUD 가 고장 상태를 읽어 갈 수 있게
    n = int(sample_rate * chunk_seconds)
    with sd.InputStream(samplerate=sample_rate, channels=num_channels,
                        dtype="float32", device=device) as stream:
        while True:
            data, overflowed = stream.read(n)          # (n, num_channels)
            if overflowed:                # 버리면 샘플 유실이 조용히 지나간다 —
                watch.overflows += 1      # 그 구간 사이렌은 그냥 사라진다
                if watch.overflows in (1, 10, 100, 1000):
                    print(f"[audio] 경고: 입력 overflow 누적 {watch.overflows}회 — "
                          "샘플 유실 가능(소비자가 느림)", file=sys.stderr)
            mono = data[:, channel].copy()
            warn = watch.update(mono)
            if warn:
                print(warn, file=sys.stderr)
            yield AudioChunk(samples=mono, sample_rate=sample_rate)


def iter_chunks_threaded(source: Iterator, maxsize: int = 120) -> Iterator:
    """source(청크 제너레이터)를 백그라운드 스레드에서 읽어 큐로 흘려보낸다.

    소비자가 변환(블로킹)하는 동안에도 캡처 스레드는 계속 마이크를 읽으므로
    '변환 중 입력 못 받음'을 막는다. 큐가 maxsize(기본 ~2분) 차면 producer 가 대기.
    """
    import threading
    import queue

    q: "queue.Queue" = queue.Queue(maxsize=maxsize)
    sentinel = object()

    def _producer():
        try:
            for item in source:
                q.put(item)
        except Exception as e:          # 캡처 에러를 소비자 쪽으로 전달
            q.put(e)
        finally:
            q.put(sentinel)

    threading.Thread(target=_producer, daemon=True).start()
    while True:
        item = q.get()
        if item is sentinel:
            break
        if isinstance(item, Exception):
            raise item
        yield item


def _find_respeaker_index(devices=None) -> int | None:
    """입력 장치 중 이름에 'respeaker'/'seeed' 가 든 첫 장치 인덱스. 없으면 None(기본 장치).

    devices: 테스트용 주입(sd.query_devices() 형태의 dict 시퀀스). None 이면 실제 조회.
    """
    if devices is None:
        import sounddevice as sd  # 지연 import
        devices = sd.query_devices()

    for idx, dev in enumerate(devices):
        name = str(dev.get("name", "")).lower()
        if dev.get("max_input_channels", 0) >= 1 and ("respeaker" in name or "seeed" in name):
            return idx
    return None


def _resolve_input_device(
    device: int | None, channels: int, devices=None
) -> tuple[int | None, str]:
    """입력 장치 결정 + 시작 로그용 라벨.

    - device 명시 시 그대로 쓴다.
    - 미지정 + 다채널(channels>1)이면 ReSpeaker 를 이름으로 자동 탐지한다.
      (다채널 요구 = ReSpeaker 의도. 시스템 기본 장치가 다른 마이크면 ch0 무음 함정
       — 분류 conf 고정·자막 0건. → docs/stt/jetson.md 트러블슈팅)
    - 그래도 없으면 None(시스템 기본 장치) 폴백 + 라벨에 해결 힌트를 남긴다.
    """
    if devices is None:
        import sounddevice as sd  # 지연 import
        devices = sd.query_devices()

    if device is None and channels > 1:
        device = _find_respeaker_index(devices)

    if device is not None:
        name = devices[device].get("name", "?") if 0 <= device < len(devices) else "?"
        return device, f"{name} (index {device})"

    label = "시스템 기본 장치"
    if channels > 1:
        label += " — ReSpeaker 미탐지, 무음이면 --device N 지정"
    return None, label


class _MicHealth:
    """현재 캡처의 SilenceWatch 를 가리키는 자리 — HUD 가 고장을 그릴 수 있게 한다.

    캡처는 제너레이터라 청크만 흘려보내고, 고장 여부는 청크에 실을 자리가 없다.
    전역 하나를 두는 대신 배선을 뜯는 방법도 있지만, 이 값은 프로세스당 마이크가
    하나라는 사실에 기대는 것이라 여기서는 이게 정직하다.
    """

    def __init__(self) -> None:
        self._watch = None

    def attach(self, watch) -> None:
        self._watch = watch

    @property
    def faulted(self) -> bool:
        return bool(self._watch is not None and self._watch.faulted)


mic_health = _MicHealth()


class SilenceWatch:
    """연속 '디지털 무음' 감시 — 장치 오선택(ch0 무음) 함정을 즉시 드러낸다.

    update(samples)는 연속 chunks개 무음이 된 순간 한 번만 경고 문자열을 돌려주고,
    소리가 다시 들어오면 리셋돼 다음 무음 구간에서 또 한 번 경고한다.
    다채널 (n, C) 입력이면 ch0(처리채널) 기준. threshold 는 정상 환경소음 RMS
    보다 훨씬 낮게 잡아 '진짜 0에 가까운' 입력만 무음으로 본다.

    ★ 경고를 stderr 에 한 번 내는 것만으로는 부족하다. 이 제품의 사용자는 화면만
    본다(청각장애 운전자). 마이크가 죽으면 입력은 무음 → '일반 도로 소음' → 평상
    화면이라, **안전장치가 죽은 걸 알 방법이 없다**. Airacle deploy 는 같은 상황에서
    비정상 종료해 run.sh 가 재시작한다(infer_trt.py:342 "스테일 버퍼로 '멀쩡한 척'
    하지 않고"). ESA 는 감독 프로세스가 없어 종료하면 그냥 꺼지므로, 대신 고장을
    **화면에 띄운다** — faulted 가 그 상태다.
    """

    def __init__(self, threshold: float = 1e-5, chunks: int = 5,
                 fault_chunks: int = 20):
        self._threshold = threshold
        self._chunks = chunks
        self._fault_chunks = fault_chunks
        self._run = 0
        self._warned = False
        self.overflows = 0            # PortAudio overflow 누적 (샘플 유실)

    @property
    def faulted(self) -> bool:
        """입력이 fault_chunks 이상 연속 무음 — 마이크/USB 사망으로 본다."""
        return self._run >= self._fault_chunks

    def update(self, samples: np.ndarray) -> str | None:
        x = np.asarray(samples)
        mono = x[:, 0] if x.ndim == 2 else x
        rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64)))) if mono.size else 0.0
        if rms >= self._threshold:
            self._run, self._warned = 0, False
            return None
        self._run += 1
        if self._run >= self._chunks and not self._warned:
            self._warned = True
            return (f"[audio] 경고: 입력이 {self._run}청크 연속 무음 — 장치 오선택일 수 있음. "
                    "장치 목록 확인: python -c \"import sounddevice as sd; print(sd.query_devices())\" "
                    "→ --device N 지정")
        return None


def _resample(data: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """간단 선형 리샘플링(테스트용). 정밀 작업은 librosa.resample 권장."""
    duration = len(data) / src_sr
    dst_len = int(duration * dst_sr)
    x_old = np.linspace(0.0, duration, num=len(data), endpoint=False)
    x_new = np.linspace(0.0, duration, num=dst_len, endpoint=False)
    return np.interp(x_new, x_old, data).astype(np.float32)
