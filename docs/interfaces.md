# 데이터 인터페이스

공통 타입의 실제 정의는 [`core/types.py`](../core/types.py)에 있다. 모듈별 독립 API와 현재
실시간 파이프라인 API를 구분해서 사용한다.

## 공통 오디오

| 항목 | 값 | 의미 |
|---|---:|---|
| `SAMPLE_RATE` | 16000 Hz | 공통 런타임 샘플레이트 |
| `CHUNK_SECONDS` | 1.0초 | 독립 모듈의 기본 입력 단위 |
| 물리 마이크 | 4개 | ReSpeaker XVF-3000 어레이 |
| USB 오디오 채널 | 6개 | ch0 처리, ch1~4 raw, ch5 재생 참조 |

```python
AudioChunk(
    samples=np.ndarray,       # (n,) 또는 (n, channels)
    sample_rate=16000,
)
```

## 분류 결과

```python
ClassResult(
    label=SoundClass,                  # siren | horn | normal_traffic
    confidence=float,
    is_emergency=bool,
    subtype=SirenSubtype | None,       # ambulance | police | fire | unknown
    subtype_confidence=float | None,
)
```

독립 `classifier.infer()`는 사이렌일 때 차종도 함께 반환한다. 통합 파이프라인은
`subtype_probs()`를 경보 중 다수결해 더 안정적으로 표시한다. `speed_neural_dir`의 원시
speed 값은 `ClassResult`에 섞지 않고 통합 진단 `info`에만 남긴다.

## 방향 결과

```python
DirectionResult(
    direction=Direction,       # front | rear | left | right | unknown
    angle_deg=float | None,
)
```

## 접근 원시 결과

```python
ApproachResult(
    motion=Motion,                         # 음량 기반 원시 판단
    speed_level=int | None,                # 음량 변화 강도, 차량 속도 아님
    proximity=str | None,                  # 최근접 | 근거리 | 원거리
    rel_distance=float | None,             # 이벤트 내 상대 거리비
    gauge=float | None,                    # 상대 근접 게이지 0~1
    energy_slope=float | None,
    frequency_slope=float | None,
    doppler_confidence=float | None,
    doppler_motion=Motion,
    tone_ratio=float | None,
    frequency_r2=float | None,
)
```

`ApproachResult.motion`은 음량 기반 원시 판단이다. 화면에 사용되는 최종 움직임은
`speed_neural_dir`·음량·직접 도플러를 `ConditionalMotionFusion`으로 합친 결과다.

## STT 결과

```python
SpeechResult(
    text=str,
    is_speech=bool,
    confidence=float,
    lang=str | None,
)
```

신뢰도가 기준보다 낮거나 음성이 아니면 `text=""`다. 긴급 상태에서는 새 STT 입력을 보내지
않고 긴급 진입 전 자막도 무효화한다.

## 독립 모듈 API

| 모듈 | 함수 | 입력 → 출력 |
|---|---|---|
| 분류 | `classifier.infer` | `AudioChunk → ClassResult` |
| 방향 | `doa.estimate_direction` | `AudioChunk → DirectionResult` |
| 접근 | `ApproachDetector.update` | `AudioChunk → ApproachResult` |
| STT | `Transcriber.transcribe` | `AudioChunk → SpeechResult` |

## 실시간 통합 API

```python
event, info = Pipeline(...).process(chunk)
```

`event`는 `pipeline.alert.AlertEvent`다.

```python
AlertEvent(
    level=str,          # NONE | PRE | WARN | CRITICAL
    kind=str,           # none | siren | horn
    label=str,
    margin=float,
    onset=bool,
    remind=bool,
    clear=bool,
    subtype=str | None,
    risk=str | None,
)
```

`info`는 화면과 진단을 위한 dict이며 주요 키는 다음과 같다.

| 키 | 의미 |
|---|---|
| `direction`, `angle` | 최종 방향과 각도 |
| `motion`, `risk` | 융합 움직임과 경보 문구 |
| `gauge`, `proximity` | 이벤트 내 상대 근접 정보 |
| `fusion_source` | 이번 움직임 판단에 선택된 증거 |
| `model_motion`, `model_confidence` | 움직임 모델 진단값 |
| `model_speed_kmh` | 모델 원시값, 제품 km/h 아님 |
| `energy_slope` | 음량 추세 |
| `frequency_slope`, `doppler_confidence` | 직접 도플러 진단값 |
| `speech` | 새로 완성된 자막 또는 `None` |
| `stt_status` | 워커 생존·큐·드롭·reset·마지막 오류 |

현재 `main.py`는 이 `AlertEvent + info`를 직접 사용해 상태기계 이벤트와 진단 정보를 보존한다.
