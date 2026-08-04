# Jetson STT 배포

대상은 Jetson Orin Nano와 ReSpeaker USB 6채널 입력이다. STT는 ch0를 사용한다.

## 핵심 구조

```text
ONNX 검출·차종·움직임 → ONNX Runtime TensorRT/CUDA → Jetson GPU
Whisper STT            → CTranslate2 CUDA          → 같은 Jetson GPU
Silero VAD             → ONNX Runtime CPU provider
```

TensorRT와 CTranslate2는 같은 GPU와 통합 메모리를 공유한다. TensorRT가 동작해도 STT의
CTranslate2가 CPU 빌드면 Whisper는 CPU에서 실행된다.

## 설치 원칙

1. `python3 -m venv`를 사용한다.
2. JetPack 버전에 맞는 NVIDIA ONNX Runtime/TensorRT 빌드를 먼저 설치한다.
3. 일반 `onnxruntime` PyPI 패키지로 NVIDIA 빌드를 덮어쓰지 않는다.
4. JetPack/CUDA 버전에 맞는 CUDA 지원 CTranslate2를 사용한다.
5. `faster-whisper`를 설치한 뒤 CPU CTranslate2가 CUDA 빌드를 다시 덮어쓰지 않았는지 확인한다.
6. numpy는 설치한 NVIDIA 휠의 ABI 요구사항에 맞춘다.

JetPack에 따라 호환 휠과 컨테이너가 달라지므로 특정 외부 인덱스 URL을 이 저장소 문서에
고정하지 않는다. 설치 후 아래 검증 결과를 통과하는지가 기준이다.

## 시스템 패키지

```bash
sudo apt update
sudo apt install -y libportaudio2 ffmpeg libusb-1.0-0

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

`libusb`는 방향 LED나 ReSpeaker 자체 DoA를 사용할 때 필요하다. 오디오 입력만 사용할 때는
`libportaudio2`가 핵심이다.

## 패키지 충돌 방지

Jetson용 `onnxruntime-gpu`가 이미 설치된 통합 환경에서는 `faster-whisper`가 일반
`onnxruntime`을 의존성으로 다시 설치하지 않게 관리한다. 필요한 STT 패키지를 수동 설치할
경우의 구성은 다음과 같다.

```bash
pip install --no-deps faster-whisper
pip install huggingface_hub tokenizers av tqdm soundfile sounddevice webrtcvad-wheels
```

이 명령만으로 CUDA CTranslate2나 NVIDIA ONNX Runtime이 설치되는 것은 아니다. 두 바이너리는
사용 중인 JetPack/CUDA에 맞는 빌드를 별도로 준비해야 한다.

## 반드시 확인할 항목

ONNX Runtime provider:

```bash
python -c "import onnxruntime as o; print(o.get_available_providers())"
```

통합 권장 결과:

```text
TensorrtExecutionProvider
CUDAExecutionProvider
CPUExecutionProvider
```

Silero VAD가 `CPUExecutionProvider`를 사용하므로 TensorRT/CUDA만 있고 CPU provider가 빠진
특수 빌드는 사용할 수 없다.

CTranslate2 CUDA:

```bash
python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count())"
```

`1` 이상이어야 한다. 마지막으로 실제 로그를 확인한다.

```bash
python -m stt.run --mic --respeaker --model small
```

```text
[stt] VAD: silero-onnx
[stt] 엔진 로드: small on cuda/float16
```

## 통합 실행

처음에는 GPU 메모리 경쟁을 줄이기 위해 `small`로 시작한다.

```bash
python -u main.py --mic --channels 6 --stt --stt-model small
```

확인할 것:

1. 시작 로그의 입력 장치가 ReSpeaker 6채널인지 확인한다.
2. STT 로그가 `cuda/float16`인지 확인한다.
3. TensorRT provider가 실제 목록에 있는지 확인한다.
4. 자막 지연, `dropped_chunks`, GPU 메모리, 사이렌 경보 지연을 함께 측정한다.
5. 여유가 있을 때만 `medium`으로 올린다.

```bash
python -u main.py --mic --channels 6 --stt --stt-model medium --view dashboard
```

## CPU 폴백

GPU STT가 준비되지 않았을 때 기능 확인은 가능하다.

```bash
python -m stt.run --wav path/to/speech.wav --model small --cpu
python main.py --mic --channels 6 --stt --stt-model small
```

통합 `main.py`는 장치를 자동 선택하므로 CUDA CTranslate2가 없으면 CPU/int8로 폴백한다.
CPU에서는 자막이 늦을 수 있지만 STT 워커가 백그라운드로 돌아 긴급음 검출은 계속된다.

## 문제 해결

| 증상 | 원인과 조치 |
|---|---|
| STT 로그가 `cpu/int8` | CUDA CTranslate2 미설치 또는 CPU 빌드가 덮어씀 |
| TensorRT provider 없음 | 일반 CPU ONNX Runtime이 설치됐거나 Jetson 빌드 불일치 |
| Silero가 WebRTC/energy로 폴백 | ONNX Runtime import와 CPU provider 확인 |
| `not compiled with CUDA support` | CTranslate2가 CPU 빌드임 |
| `illegal memory access`/OOM | STT를 `small`로 낮추고 다른 GPU 프로세스와 메모리 확인 |
| STT 입력 생략 누적 | 디코딩이 실시간 입력보다 느림. 모델 축소 또는 CUDA 경로 수정 |
| 자막이 나오지만 약 5초 늦음 | 로그가 CPU인지 확인. GPU여도 실제 Jetson 지연 측정 필요 |
| 차종·접근 GPU가 갑자기 CPU로 바뀜 | 일반 onnxruntime이 NVIDIA 빌드를 덮어썼는지 확인 |

전체 실행 명령은 [공통 실행 문서](../running.md)에 있다.
