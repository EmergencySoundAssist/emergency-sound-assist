# STT(음성→텍스트) 설계  *(확장 기능 — 담당: 천자민)*

> 목표: 오디오 → **텍스트** → `core.types.SpeechResult` 반환
> 구현 파일: [`stt/transcriber.py`](../../stt/transcriber.py) — `Transcriber.transcribe(chunk) → SpeechResult`

청각장애 운전자는 사이렌·경적뿐 아니라 **사람의 말**도 못 듣는다.
경찰 확성기("차 세우세요"), 옆 차/행인의 외침, 안내방송 등을
**텍스트로 바꿔 보여 주면** 상황을 더 빨리 알 수 있다.

분류(siren/horn, ① 모듈) 와 달리 STT 는 **말의 내용**을 텍스트로 옮길 뿐이다.
(긴급 여부 판단은 ① 분류 모듈의 몫 — STT 는 텍스트만 책임진다.)

---

## 처리 흐름

```
AudioChunk(1초) 흐름
   │
   ├─ 무음 게이트(VAD): RMS 에너지 < 임계 → 버림 (엔진 안 돌림, 비용 절약)
   │
   └─ 음성이면 버퍼에 누적 ──┐
        · 무음이 이어지면(발화 끝) │ → 모은 발화를 STT 엔진에 한 번에 인식
        · 또는 최대 길이 초과 ─────┘
                  │
                  ▼
        텍스트 → SpeechResult
```

- **왜 버퍼링?** 1초 청크는 STT 에 너무 짧다. 음성이 이어지는 동안 모았다가
  **발화 단위**로 인식해야 문장이 제대로 나온다. → `ApproachDetector` 처럼 상태를 가진 클래스.
- **왜 VAD(무음 게이트)?** 도로는 대부분 '말 없는' 구간. 조용할 때 엔진을 안 돌리면 연산을 크게 아낀다.

---

## 2단계 전략

### 1단계 (MVP): faster-whisper + 에너지 VAD
- 엔진: **faster-whisper**(CTranslate2 기반) — 오프라인·CPU 동작·한국어 지원·Jetson 이식 가능.
- VAD: 청크 RMS 에너지 임계값(간단·의존성 0). `config.vad_rms_threshold`.
- 인식 품질: 발화 단위 버퍼링 + Whisper 전 RMS 정규화 + 환각 가드 임계값.
- 장점: 의존성 한 개, 코드 짧음. 한계: 도로 소음에서 인식률·VAD 민감도 튜닝 필요.

### 2단계 (개선, 필요시)
- **VAD 고도화**: 에너지 임계 → webrtcvad / Silero VAD 로 교체(노이즈에 강함).
- **스트리밍 인식**: 발화 끝까지 기다리지 않고 부분 결과를 더 빨리 표시.
- **Jetson 가속**: `device="auto"` 가 CUDA 자동 감지(cuda/float16). 배포·의존성 충돌 → [jetson.md](jetson.md).
- **DoA 연계**: ② 방향 결과와 합쳐 "후방에서 누군가 말함" 처럼 말의 **방향**까지 표시.

> ⚠️ **먼저 1단계로 인식·VAD 품질 측정 → 부족하면 2단계.** (과잉설계 방지)

---

## 엔진 교체 포인트
`Transcriber(engine=...)` 로 엔진을 주입할 수 있다. `_Engine` 프로토콜
(`transcribe(samples, sr) → (text, confidence, lang)`)만 지키면
faster-whisper 를 vosk 등으로 바꿔도 나머지 코드는 그대로다.
테스트는 가짜 엔진을 주입해 **faster-whisper 없이** 돌아간다.

---

## 설정 (`stt/config.py`)
| 항목 | 기본값 | 의미 |
|------|--------|------|
| `model_size` | small | tiny/base/small… (클수록 정확·느림). 한국어는 small 최소 |
| `language` | ko | None 이면 자동 감지 |
| `device` / `compute_type` | auto | 노트북=cpu/int8, Jetson(GPU)=cuda/float16 자동 |
| `vad_rms_threshold` | 0.005 | 이 이상 RMS 면 음성으로 간주(낮춰서 범위↑) |
| `normalize_audio` | True | Whisper 전 RMS 정규화(먼/조용한 음성 살림) |
| `max_utterance_seconds` | 8.0 | 한 발화 최대 길이(넘으면 강제 인식) |
| `silence_release_chunks` | 1 | 음성 뒤 무음이 이만큼 연속이면 발화 끝 |
| `min_utterance_seconds` | 0.5 | 이보다 짧으면 잡음으로 버림 |

---

## 실행 / 검증
```bash
pip install -r stt/requirements.txt
python -m stt.run --wav some_speech.wav      # 파일 한 방 인식
python -m stt.run --mic                       # 노트북 마이크 실시간
pytest tests/test_stt.py                       # 엔진 없이 로직 검증(가짜 엔진)
```

## 참고
- 출력 형식: [../interfaces.md](../interfaces.md) (`SpeechResult`)
- **Jetson 배포 + 의존성 충돌 정리**: [jetson.md](jetson.md)
- 입력/오디오 공통: [../architecture.md](../architecture.md)
- 평가: 알려진 문장을 도로 소음과 섞어 재생 → 인식 정확도(WER) 표.

## TODO
- [x] 인터페이스(`SpeechResult`) + 모듈 스켈레톤 + 단위 테스트
- [x] faster-whisper 엔진 래퍼(지연 import) + 에너지 VAD + 발화 버퍼링
- [x] 인식 품질: RMS 정규화 + 환각 가드 + 라이브 상태 표시(미터/변환중)
- [ ] 실제 faster-whisper 로 WAV 인식 품질 측정(노트북)
- [ ] 도로 소음 환경에서 VAD 임계값 튜닝
- [ ] (필요시) Silero VAD / 스트리밍 / Jetson 가속
- [ ] (선택) DoA 와 연계해 말의 방향까지 표시
