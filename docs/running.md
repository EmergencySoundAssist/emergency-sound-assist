# 실행 명령어

모든 명령은 저장소 루트에서 실행한다.

## 1. 노트북 개발 환경

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

STT까지 사용할 때만 추가한다. Whisper 모델은 첫 실행 때 내려받으므로 인터넷 연결이 필요하다.

```bash
pip install -r stt/requirements.txt
```

Jetson은 일반 PyPI 패키지가 CUDA/TensorRT 빌드를 덮어쓸 수 있으므로 위 설치 명령을 그대로
사용하지 않는다. [Jetson 통합 확인](#8-jetson-gpu--tensorrt-확인)과 각 모듈 런북을 먼저 본다.

## 2. 가장 먼저 실행할 명령

하드웨어 없이 전체 경보 파이프라인을 확인한다.

```bash
python main.py --demo
python main.py --demo --view dashboard
```

WAV 파일은 모노 경로이므로 분류·접근은 동작하지만 방향은 `미상`이다.

```bash
python main.py --wav path/to/audio.wav
```

ReSpeaker 통합 실행은 USB 6채널로 연다.

```bash
python main.py --mic --channels 6
python main.py --mic --channels 6 --view dashboard
python main.py --mic --channels 6 --device 2
```

평상시 음성 자막까지 켠다. CPU 지연을 줄일 때는 `small`, 정확도 우선 기본값은 `medium`이다.

```bash
python main.py --mic --channels 6 --stt --stt-model small
python main.py --mic --channels 6 --stt --stt-model medium --view dashboard
```

통합 실행에서 `ch0`은 분류·차종·접근·STT, `ch1~4`는 방향에 사용하고 `ch5`는 사용하지 않는다.

감지 결과를 워치 진동으로 보내려면 `--ble`을 켠다. 자세한 내용은 아래 BLE 절을 본다.

```bash
python main.py --mic --channels 6 --ble
python main.py --mic --channels 6 --stt --stt-model small --ble --view dashboard
```

## 3. 입력 장치 확인

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

ReSpeaker가 자동으로 잡히지 않으면 입력 채널이 6개 이상인 장치 번호를 `--device N`으로 준다.

## 4. 소리 분류만 실행

```bash
python classify.py --demo
python classify.py --wav path/to/audio.wav
python classify.py --mic
python classify.py --mic --channels 6 --device 2
```

분류기는 5초 롤링 창을 사용하므로 시작 직후에는 버퍼 워밍업 구간이다. 통합 파이프라인은
별도의 2초 모델로 PRE 예비경보를 먼저 만들고 5초 모델로 확정한다.

## 5. 방향 추정만 실행

하드웨어 없는 합성 검증:

```bash
python -m doa.multi_live --demo
python -m doa.multi_live --demo --algo MUSIC
```

ReSpeaker 실시간 방향:

```bash
python -m doa.multi_live --no-led
python -m doa.multi_live --led
python -m doa.multi_live --device 2 --no-led
```

장착 방향 보정:

```bash
python -m doa.diag --truth front
python -m doa.diag --truth rear
python -m doa.diag --truth left
python -m doa.diag --truth right
```

세부 옵션은 [DoA 실행 문서](doa/running.md)를 참고한다.

## 6. STT만 실행

```bash
python -m stt.run --wav path/to/speech.wav
python -m stt.run --mic --model small
python -m stt.run --mic --respeaker --model small
python -m stt.run --mic --respeaker --model medium --accuracy
```

`--accuracy`는 beam을 늘려 더 느리다. VAD를 비교할 때는 다음처럼 강제할 수 있다.

```bash
python -m stt.run --mic --vad-backend silero
python -m stt.run --mic --vad-backend webrtc
python -m stt.run --mic --vad 0.02       # energy backend로 자동 전환
```

## 7. 테스트와 공개 데이터 평가

```bash
python -m pytest -q
```

접근·멀어짐 융합 재현:

```bash
pip install -r evaluation/requirements.txt
python -m evaluation.download_figshare_samples --kind siren --count 12
python -m evaluation.benchmark_motion --sources 10 \
  --output data/evaluation_results_10sources.json
python -m evaluation.compare_methods_cv data/evaluation_results_10sources.json
python -m evaluation.evaluate_speed_output data/evaluation_results_10sources.json
```

STT 재현:

```bash
python -m evaluation.download_figshare_samples --kind road --count 6
python -m evaluation.download_fleurs_samples --count 20
python -m evaluation.benchmark_stt --corpus fleurs --model medium \
  --snr clean,20,10,5 --output data/stt_evaluation_fleurs_medium.json
python -m evaluation.benchmark_stt --corpus tts --model medium \
  --snr clean,20,10,5 --output data/stt_evaluation_tts_medium.json
```

내려받은 데이터와 결과는 `data/`에 저장되며 Git에는 포함하지 않는다.

## 8. Jetson GPU · TensorRT 확인

TensorRT와 CTranslate2는 서로 다른 장치가 아니라 같은 Jetson GPU를 사용하는 실행 엔진이다.

ONNX 모델의 실행 provider를 확인한다.

```bash
python -c "import onnxruntime as o; print(o.get_available_providers())"
```

권장 출력에는 다음 provider가 포함된다.

```text
TensorrtExecutionProvider
CUDAExecutionProvider
CPUExecutionProvider
```

STT용 CTranslate2 CUDA를 별도로 확인한다.

```bash
python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count())"
```

`1` 이상이어야 하며 실행 로그는 다음처럼 나와야 한다.

```text
[stt] 엔진 로드: small on cuda/float16
```

`cpu/int8`이면 ONNX 모델이 TensorRT를 사용하더라도 STT만 CPU로 실행되는 상태다. 자세한 설치
주의사항은 [STT Jetson 런북](stt/jetson.md), 방향 장치 설정은 [DoA Jetson 런북](doa/jetson.md)을 본다.

## 9. 자주 발생하는 문제

| 증상 | 확인할 것 |
|---|---|
| 방향이 계속 미상 | `--channels 6`, ReSpeaker 장치 번호, ch1~4 입력 |
| 자막이 약 5초 뒤 표시 | STT 로그가 `cuda/float16`인지 확인하고, CPU면 `small` 사용 |
| STT 입력 생략 경고 증가 | 모델 디코딩이 입력보다 느림. 모델 축소 또는 CUDA 활성화 |
| `Invalid number of channels` | 선택한 장치가 6채널 입력을 지원하는지 확인 |
| TensorRT가 목록에 없음 | Jetson용 ONNX Runtime/TensorRT 빌드가 아닌 CPU 패키지일 수 있음 |
| 차종이 표시되지 않음 | 사이렌 확정 여부와 `subtype_cnn_attn_yt_s42.onnx` 존재 확인 |
| 시작 직후 결과가 불안정 | 검출·차종·움직임 모델의 5초 창이 아직 채워지는 중 |
| `[ble] bleak이 없습니다` | `pip install bleak` |
| BLE가 계속 재연결만 반복 | 폰 GATT 서버가 켜져 있는지, 서비스 UUID가 같은지 확인 |

## 10. BLE 전송 (`--ble`)

Jetson이 BLE 클라이언트(중앙기기)로 동작해 폰(GATT 서버)에 4바이트를 write 하고, 폰이 워치로
미러링해 진동시킨다. 워치에 직접 연결하지 않는다.

```bash
sudo apt install -y bluez
pip install bleak
```

폰의 GATT 서버를 서비스 UUID로 자동 검색한다.

```bash
python main.py --mic --channels 6 --ble
```

자동 검색이 느리거나 실패하면 MAC을 직접 준다. 더 빠르고 확실하다.

```bash
python main.py --mic --channels 6 --ble --watch-mac AA:BB:CC:DD:EE:FF
```

먼저 하드웨어 없이 전송 경로만 확인할 수 있다.

```bash
python main.py --demo --ble
```

페이로드는 4바이트 고정이며 `notify/protocol.py`와 워치쪽 `AlertProtocol.kt`가 **반드시 같아야
한다**. 한쪽만 바꾸면 워치가 값을 잘못 읽는다.

| 바이트 | 의미 | 값 |
|---|---|---|
| 0 | 소리 | 0=일반, 1=사이렌, 2=경적 |
| 1 | 방향 | 0=전방, 1=후방, 2=좌, 3=우, 0xFF=미상 |
| 2 | 움직임 | 0=접근, 1=멀어짐, 2=유지, 0xFF=미상 |
| 3 | 신뢰도 | 0~100 |

동작 특성:

- 소리 종류는 매 tick 원시 분류가 아니라 **디바운스된 경보 상태기계**(ONSET/REMIND/CLEAR)를
  따른다. 사이렌 유지 중에는 값이 안정적이고, 해제되면 `0`을 보내 폰이 진동을 멈춘다.
- 직전과 같은 페이로드는 보내지 않는다(진동·트래픽 폭주 방지).
- 큐는 최신 1건만 유지하고 Write Without Response를 쓴다(지연 최소화).
- 연결은 시작 시 1회 맺고 끊기면 자동 재연결한다. 매 전송마다 스캔하지 않는다.
- BLE 실패가 감지 파이프라인을 멈추지 않는다. 전송만 조용히 생략된다.
- 움직임 바이트는 조건부 융합 결과다. **속도 단계는 보내지 않는다** —
  공개 데이터 검증에서 기준선 이하였다([검증 문서](approach/validation.md)).
