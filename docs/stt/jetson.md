# Jetson 배포 런북 (STT) + 의존성 충돌 정리

대상: **Jetson Orin Nano Super Dev Kit** (JetPack 6.x / L4T r36.x, aarch64, CUDA 12.x, cuDNN 9, GPU SM 8.7, 통합 8GB).
입력은 ReSpeaker **ch0**(빔포밍된 깨끗한 모노) → `python -m stt.run --mic --respeaker`.

> ⚠️ **conda 금지.** aarch64/JetPack 에선 Anaconda 가 맞지 않는다 — `python3 -m venv` 사용 (doa 런북과 동일).
> 핵심 원칙: **런타임(보드)과 학습(노트북)을 분리한다.** torch/tensorflow 는 학습용이라 보드에 올리지 않는다.

---

## 0. 먼저 — 왜 그냥 `pip install` 하면 안 되나 (브랜치 교차 충돌)

이 프로젝트는 한 보드에 여러 모듈이 **같은 venv** 로 올라간다. 각 브랜치 의존성:

| 브랜치 | 핵심 의존성 | Jetson 런타임? |
|--------|------------|:----:|
| `feature/classifier` | torch, torchvision, torchaudio, **tensorflow**, tf-hub, pandas, scikit-learn | ❌ **학습 전용** |
| `feat/realtime-approach-pipeline` | numpy, scipy, **onnxruntime**(보드=onnxruntime-gpu), soundfile, sounddevice | ✅ 런타임 |
| `doa-jamin` | numpy, scipy, pyroomacoustics, **pyusb**, sounddevice | ✅ 런타임 |
| `feat/stt` (이 모듈) | **faster-whisper**(→ ctranslate2, onnxruntime, av, tokenizers, hf-hub), numpy, soundfile, sounddevice | ✅ 런타임 |

**충돌 4가지 (실제로 보드를 깨뜨리는 것들):**

1. **`onnxruntime` 덮어쓰기 (가장 위험).**
   `faster-whisper` 는 PyPI `onnxruntime`(CPU) 을 **강제 의존**(`onnxruntime<2,>=1.14`)한다.
   approach 는 Jetson 에서 **`onnxruntime-gpu`**(NVIDIA aarch64 전용 빌드)를 쓰는데, 둘 다 같은
   `onnxruntime` 파이썬 네임스페이스에 설치돼 **서로를 덮어쓴다** → approach 의 GPU/TensorRT 추론이 깨진다.
   ✅ **해결**: 우리 STT 는 **자체 VAD**(webrtcvad, 없으면 energy 폴백)를 쓰므로 faster-whisper 의 onnxruntime(Silero VAD)이 **필요 없다**.
      → `pip install --no-deps faster-whisper` 로 onnxruntime 을 끌어오지 않게 한다.

2. **ctranslate2 CPU 휠이 CUDA 빌드를 덮어쓰기.**
   보드에서 CUDA 가속 ctranslate2 를 깔아도, 이후 `pip install`(의존성 재해결)이 **PyPI CPU 휠**을
   조용히 다시 덮어쓴다 → 런타임에 `device='cuda'` 가 "not compiled with CUDA support" 로 죽는다.
   ✅ **해결**: **CUDA ctranslate2 를 맨 마지막에** 설치하고, 이후 dep 재해결을 피한다. faster-whisper 는 `--no-deps`.

3. **numpy 2.x ABI 깨짐.**
   Jetson 의 NVIDIA 전용 휠(torch, onnxruntime-gpu)은 보통 **numpy 1.x** 로 빌드된다. numpy 2.x 가 섞이면
   ABI 불일치로 import 시 깨진다. (노트북은 numpy 2.3.x 라도 보드는 다름)
   ✅ **해결**: 보드에선 **numpy 를 NVIDIA 휠에 맞춰 핀**(보통 `numpy<2`, 예: `numpy==1.26.4`). 모든 브랜치가
      `numpy>=1.24` 라 1.26 으로 통일 가능. ctranslate2 는 numpy 핀이 없어 따라온다.

4. **torch/tensorflow 를 보드에 올리는 것 자체.**
   tensorflow 는 aarch64/Python 3.10 에서 설치가 고통스럽고, 런타임에 필요도 없다.
   ✅ **해결**: **학습은 노트북, 보드는 ONNX/CTranslate2 추론만.** (approach 가 이미 이 구조)

---

## 1. 시스템 패키지 (apt)

```bash
sudo apt update
sudo apt install -y libportaudio2 ffmpeg            # sounddevice 런타임 + av(PyAV) 디코딩
sudo apt install -y libusb-1.0-0                    # (doa LED/XVF 쓸 때만) pyusb 백엔드
# cuDNN 9 / CUDA 12.x 는 JetPack 에 이미 포함 — 따로 설치하지 말 것 (cuDNN 8 설치 금지)
sudo nvpmodel -m 0 && sudo jetson_clocks            # MAXN + 클럭 고정(실시간 지연 최소화)
```

## 2. 가상환경 + 공통 런타임 의존성 (CUDA ctranslate2 제외)

```bash
cd ~/emergency-sound-assist
python3 -m venv .venv && source .venv/bin/activate && pip install -U pip

# numpy 를 먼저 핀(다른 휠이 2.x 로 끌어올리지 못하게):
pip install "numpy==1.26.4"

# STT 비-충돌 의존성만 (onnxruntime/ctranslate2 는 제외 — 아래 GPU 단계에서):
pip install --no-deps faster-whisper
pip install huggingface_hub tokenizers av tqdm soundfile sounddevice webrtcvad-wheels
# approach 의 onnxruntime-gpu 는 그쪽 런북대로 NVIDIA 휠로 설치(STT 가 건드리지 않음)
```

## 3. CUDA ctranslate2 — 셋 중 하나 (A 권장)

> PyPI 의 aarch64 ctranslate2 휠은 **CPU 전용**이다. GPU 를 쓰려면 아래 중 하나로 **CUDA 빌드**를 얻어야 한다.
> (mid-2026 검증: 단일 bare-metal pip 경로는 보드 CUDA 버전에 따라 불안정 → **Docker 가 가장 확실**)

**옵션 A (권장) — dusty-nv jetson-containers / 턴키 이미지**
```bash
git clone https://github.com/dusty-nv/jetson-containers
bash jetson-containers/install.sh
jetson-containers run $(autotag faster-whisper)     # CUDA ctranslate2 + faster-whisper 내장
# 또는 OpenAI 호환 STT 서버 이미지:
# docker run -d --runtime nvidia -p 8000:8000 \
#   -v speaches-models:/home/ubuntu/.cache/huggingface/hub \
#   cbinckly/speaches:0.9.0-l4t-cuda-12.6.11-arch87
```

**옵션 B — Jetson 휠 인덱스 (bare-metal, .venv 안에)**
```bash
# ⚠️ jp6/cu126 인덱스는 ctranslate2 가 내려갔다(2025-10). 살아있는 건 jp6/cu129.
#    이 휠은 CUDA 12.9 빌드라, stock JetPack 6.2(CUDA 12.6)면 cuda-compat/12.9 툴킷 먼저 필요.
pip install --index-url https://pypi.jetson-ai-lab.io/jp6/cu129 ctranslate2   # 맨 마지막에!
python -c "import ctranslate2; print('cuda devices:', ctranslate2.get_cuda_device_count())"  # >=1 이어야 함
# 'libctranslate2.so.4: cannot open shared object file' 면 → 옵션 A 또는 C 로.
```

**옵션 C (최후수단, 확실) — 소스 빌드 (SM 8.7)**
```bash
sudo apt install -y libcudnn9-dev-cuda-12 libopenblas-dev build-essential cmake git
sudo fallocate -l 8G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile  # OOM 방지
git clone --recursive https://github.com/OpenNMT/CTranslate2
cmake -S CTranslate2 -B CTranslate2/build \
  -DWITH_CUDA=ON -DWITH_CUDNN=ON -DWITH_MKL=OFF -DWITH_OPENBLAS=ON \
  -DOPENMP_RUNTIME=COMP -DCMAKE_CUDA_ARCHITECTURES=87 -DCMAKE_BUILD_TYPE=Release
cmake --build CTranslate2/build -j2          # -j2 (NOT -jnproc) — 8GB OOM 방지
sudo cmake --install CTranslate2/build && sudo ldconfig
( cd CTranslate2/python && pip install -r install_requirements.txt && python setup.py bdist_wheel )
pip install CTranslate2/python/dist/*.whl
python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count())"   # >=1
```

## 4. CPU 폴백 (GPU 없이 파이프라인 검증 — 가장 간단)

```bash
cd ~/emergency-sound-assist && python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install "numpy==1.26.4" faster-whisper soundfile sounddevice
python -m stt.run --wav some_speech.wav     # CPU/int8 로 동작 (resolve_runtime 이 auto 로 cpu 선택)
```
- aarch64 PyPI 휠로 **CPU 추론은 그냥 된다.** `tiny`/`base` 는 실시간 가능, `small` 은 경계선.
- 실시간 한국어는 GPU 경로 권장.

## 5. 실행 + 런타임 설정

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"   # 6채널 'ReSpeaker' 인덱스 확인
python -m stt.run --mic --respeaker            # ch0 자동, GPU 있으면 자동 사용
python -m stt.run --mic --respeaker --device 2 # 자동 탐지 실패 시 인덱스 지정
```
- 코드 기본값(`device="auto"`)이 `ctranslate2.get_cuda_device_count()` 로 **cuda/cpu 자동 선택** → 노트북·보드 동일 코드.
- 보드 권장 프로파일: `STTConfig.for_jetson()` = **small + cuda + float16** (Orin Tensor 코어 100% 활용, 최상급 정확도). 8GB 가 빡빡하면 `int8_float16` 으로.
  한국어는 Whisper 난이도가 높아 `base` 보다 `small` 이 인식 정확도에 안정적.

## 5.5 통합 실행 (main.py — 긴급↔STT 게이트 실증)

```bash
python -u main.py --mic --channels 6 --stt --stt-model small
```
체크 포인트 (위에서부터 순서대로 확인):
1. `[audio] 입력 장치: ReSpeaker …` — **다른 장치명이 찍히면** `python -c "import sounddevice as sd; print(sd.query_devices())"` 로 6채널 장치 인덱스 확인 후 `--device N`.
2. `[stt] VAD: webrtcvad` — `energy 폴백` 이 찍히면 `pip install webrtcvad-wheels`.
3. CUDA 미감지(CPU) 상태에선 **반드시 `--stt-model small`** — medium(기본값)/turbo 는 STT 워커가 못 따라와 자막이 안 뜬다. GPU ctranslate2(3번) 설치 후에는 기본값으로 가능.
4. `| tee` 등 파이프로 로그를 뜰 때는 `python -u`(stdout 언버퍼) 필수 — 안 그러면 진행 줄이 한참 안 보인다.
5. `[audio] 경고: … 연속 무음` 이 뜨면 ch0 에 소리가 안 들어오는 상태(장치 오선택) — 1번으로.

## 5.6 HUD 화면 (Jetson 직결 디스플레이)

```bash
sudo apt install fonts-nanum          # 한글 폰트(없으면 □□로 깨짐)
pip install "pygame>=2.1"             # 보드 venv에 설치(aarch64 휠)
python -u main.py --mic --channels 6 --stt --stt-model small --hud
```
- 긴급(사이렌·경적): 방향 레이더에 방향 섹터 점등 + 소리·접근 표시. 평상시: 하단 자막 밴드.
- 종료 `ESC`/`Q`/창 닫기 · 반사(윈드실드 상하반전) 토글 `F` · 시작부터 반사면 `--hud-flip`.
- 노트북 개발은 `--hud-windowed`(창 모드).

### 부팅 자동시작

```bash
sudo cp deploy/emergency-hud.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now emergency-hud.service
journalctl -u emergency-hud -f        # 로그 확인
```
- 유닛의 `User`·경로·`DISPLAY`/`XAUTHORITY`를 보드 계정(예: dcnm)·그래픽 세션에 맞춘다.
- `pygame.error: ... video system` / 권한 거부면 로그인 세션에서 `xhost +SI:localuser:dcnm` 후 재시작.

## 6. 트러블슈팅

| 증상 | 원인 / 해결 |
|------|------------|
| 분류 conf 가 고정(예: 0.83)·자막 0건 | 입력 장치 오선택 → ch0 무음. 시작 로그 `[audio] 입력 장치` 확인, `--device N` 지정. |
| `not compiled with CUDA support` | CPU 휠이 깔림. CUDA ctranslate2(3번) 재설치, `get_cuda_device_count()>=1` 확인. |
| approach 의 GPU 추론이 깨짐 | faster-whisper 가 `onnxruntime`(CPU) 으로 onnxruntime-gpu 를 덮음. STT 는 `--no-deps` 로 설치(1번 충돌). |
| `libctranslate2.so.4: cannot open` | 파이썬 래퍼만 깔리고 C++ 런타임 없음. 옵션 A(컨테이너)/C(소스빌드)로. |
| import 시 numpy ABI 에러 | numpy 2.x 가 섞임. `pip install "numpy==1.26.4"` 로 통일(충돌 3). |
| `illegal memory access` / `CUBLAS_NOT_INITIALIZED` | 8GB 통합메모리 고갈. `small`+`int8_float16`, 콘솔 부팅(데스크탑 끔), 다른 GPU 프로세스 종료. |
| 한국어가 인식 안 됨 | `.en` 모델 쓰면 안 됨. 멀티링궐 `small`/`medium` 사용. |

## 7. (참고) systemd 상시 구동

```ini
[Service]
WorkingDirectory=/home/<user>/emergency-sound-assist
ExecStart=/home/<user>/emergency-sound-assist/.venv/bin/python -m stt.run --mic --respeaker
Restart=on-failure
```
`python -m stt.x` 는 **CWD 의존**이라 `WorkingDirectory` 를 레포 루트로 지정(core/stt import 해결).

---

## 부록 — 엔진 대안 (필요시)
- **whisper.cpp (CUDA)**: CTranslate2 빌드가 너무 고통스러우면 폴백. 같은 Whisper 가중치라 한국어 정확도 비슷, C/C++ 빌드라 자체완결. 멀티링궐 ggml 모델 사용. `_Engine` 프로토콜로 교체만 하면 됨.
- **NVIDIA Riva / Vosk**: Riva 는 진짜 스트리밍이지만 무겁고 Orin Nano 8GB 엔 과함. Vosk 는 가볍지만 한국어 WER 높아 안전-크리티컬엔 부적합.
- 결론: **faster-whisper 기본, whisper.cpp-CUDA 폴백.**

> 설치 명령·버전·핀은 mid-2026 웹 검증 기준(jetson-containers / ctranslate2 CHANGELOG / NVIDIA 포럼). 보드의 실제 JetPack/CUDA 버전을 `jtop` 으로 먼저 확인할 것.
