# ④ STT (음성→텍스트) 모듈

주변 음성(경찰 확성기·외침·안내방송)을 텍스트로 바꾸고, 운전 긴급 키워드
(구급차·비키세요·정지 등)를 짚어 청각장애 운전자에게 보여 준다.

- 인터페이스: `Transcriber.transcribe(chunk) → SpeechResult` (`core/types.py`)
- 설계: [../docs/stt/design.md](../docs/stt/design.md)
- Jetson 배포/의존성 상세: [../docs/stt/jetson.md](../docs/stt/jetson.md)

---

## 빠른 실행 (노트북 = Jetson 기본 경로, 난이도 동일)

다른 모듈과 똑같이 **그냥 설치하고 실행**하면 된다. CPU로 돈다.

```bash
pip install faster-whisper
python -m stt.run --wav some_speech.wav      # 파일 한 방 인식
python -m stt.run --mic                       # 기본 마이크 실시간
python -m stt.run --mic --respeaker           # ReSpeaker ch0 (Jetson 타겟)
python -m stt.run --mic --model small         # 한국어는 small 권장
```

- 코드가 `device="auto"` 라 **GPU가 잡히면 자동으로 CUDA**, 없으면 CPU로 떨어진다 → 노트북·Jetson 동일 코드.
- 말하고 **잠깐 멈추면** 그때 한 문장이 출력된다(무음=발화 끝 신호). **Ctrl+C** 종료.
- 엔진 없이 로직만 검증: `pytest tests/test_stt.py` (가짜 엔진 주입, 설치 불필요).

---

## "왜 STT만 Jetson 문서가 복잡해 보이나?" — 정리

결론부터: **그냥 돌리는 건 위처럼 다른 브랜치만큼 간단하다.** 복잡한 부분은
전부 *실시간 GPU 가속*과 *네 모듈을 한 보드에 합칠 때*의 이야기이고, **선택 사항**이다.

### 1. STT만 GPU를 원하는 모듈이라서 (본질적 이유)
| 모듈 | 연산 | GPU 필요? |
|------|------|:--------:|
| doa | 신호처리 (numpy/scipy/pyroomacoustics) | ❌ CPU로 충분 |
| approach | 작은 CNN (ONNX) | ❌ CPU로 충분 |
| classifier | 학습은 torch/TF, **보드 런타임은 ONNX** | ❌ (학습은 노트북) |
| **stt** | **Whisper (큰 트랜스포머)** | ⚠️ 실시간이면 GPU |

→ doa/approach는 "aarch64에서 CUDA 빌드" 벽을 만날 일이 없다. STT만 **GPU를 원하는 순간**
   그 벽을 만난다. 코드 설계 탓이 아니라 "큰 모델"이라는 본질 때문.

### 2. 하필 CTranslate2의 Jetson-CUDA 사정이 나빠서 (라이브러리 속성)
- approach의 onnxruntime은 NVIDIA가 Jetson용 `onnxruntime-gpu`를 제공 → 비교적 표준.
- faster-whisper의 백엔드 CTranslate2는 **aarch64 CUDA 휠을 깔끔히 제공하지 않음**
  → Docker(jetson-containers) 또는 소스빌드가 필요. (PyPI aarch64 휠은 CPU 전용)

### 3. 의존성 충돌은 "전 모듈 합칠 때"만, 그것도 플래그 하나
- faster-whisper가 `onnxruntime`(CPU)을 끌어와 approach의 `onnxruntime-gpu`를 덮을 수 있음.
- 우리 STT는 **자체 energy VAD**라 onnxruntime이 불필요 → `pip install --no-deps faster-whisper` 하나로 회피.
- 아직 네 모듈을 한 보드에 합친 적이 없으니 **지금 당장의 문제는 아님**(미래 대비).

### 다른 모듈도 각자 Jetson 주의사항이 있다
- `doa-jamin` → [../docs/doa/jetson.md](../docs/doa/jetson.md) (pyroomacoustics 빌드, udev 규칙)
- `feat/realtime-approach-pipeline` → requirements.txt에 onnxruntime-gpu/TensorRT 주석
- 즉 **모든 런타임 모듈이 각자 함정이 있고**, STT는 그중 "GPU를 원하는" 한 개일 뿐.

---

## 속도 올리기 (느릴 때)

영향 큰 순서:

1. **전력 모드 최대 (무료·즉시, CPU/GPU 둘 다 큼)**
   ```bash
   sudo nvpmodel -m 0 && sudo jetson_clocks
   ```
2. **GPU 가속 (가장 큰 폭)** — CPU `small`은 경계선, GPU `small`은 ~6-7배 실시간.
   → [../docs/stt/jetson.md](../docs/stt/jetson.md)의 GPU 절차(Docker 권장).
3. **CPU면 모델 줄이기** — `--model base`(small의 2~3배 빠름) 또는 `--model tiny`. 한국어 정확도와 트레이드오프.
4. **CPU 스레드 늘리기** — Orin 6코어:
   ```bash
   python -m stt.run --mic --respeaker --model base --threads 6
   ```
5. 이미 적용된 것: `beam_size=1`(greedy), `int8`/`int8_float16`, energy VAD(무음 스킵),
   발화 단위 인식, `condition_on_previous_text=False`(반복/환각 억제).

> 체감 지연(latency)은 "말 끝나고 인식"하는 발화 단위 설계 탓도 있다. 더 빨리 반응하게 하려면
> `STTConfig.silence_release_chunks`/`max_utterance_seconds`를 줄이되, 문장이 잘릴 수 있어 주의.

## 두 갈래로 보면 간단
| 목적 | 방법 | 난이도 |
|------|------|--------|
| **그냥 돌리기** | `pip install faster-whisper` + 실행 (CPU) | 다른 브랜치와 동일 |
| **실시간 GPU** | [../docs/stt/jetson.md](../docs/stt/jetson.md)의 GPU 절차 (선택 업그레이드) | Docker 권장 |
