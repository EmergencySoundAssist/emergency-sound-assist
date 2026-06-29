# 데이터 인터페이스 (팀 공통 약속)

세 모듈(분류/방향/접근)이 따로 개발돼도, 여기 형식만 지키면 `pipeline`에서 문제없이 합쳐진다.
실제 정의는 [`core/types.py`](../core/types.py)에 코드로 있다.

---

## 공통 오디오 설정
| 상수 | 값 | 의미 |
|------|-----|------|
| `SAMPLE_RATE` | 16000 | 샘플레이트 (Hz) |
| `CHUNK_SECONDS` | 1.0 | 분석 단위 길이 (초) |
| 물리 마이크 수 | 4 | ReSpeaker XVF-3000 |
| USB 채널 수 | 6 | ch0 처리됨 / ch1~4 원본 / ch5 재생 (→ [hardware.md](hardware.md)) |

> ⚠️ TODO: `core/types.py`의 `CHANNELS=4` 주석을 `NUM_MICS=4` / `USB_CHANNELS=6`으로 정정 예정.

## 입력: `AudioChunk`
```
samples: np.ndarray   # (n,) 모노 또는 (n, channels) 다채널
sample_rate: int = 16000
```
- 분류: 모노(또는 ch0) 사용 / 방향: 원본 4채널(ch1~4) 사용

---

## ① 분류 출력: `ClassResult`
```
label: SoundClass        # siren | horn | normal_traffic
confidence: float        # 0.0 ~ 1.0
is_emergency: bool        # siren/horn 이면 True
```

## ② 방향 출력: `DirectionResult`
```
direction: Direction     # front | rear | left | right | unknown
angle_deg: float | None  # 원시 각도(있으면)
```

## ③ 접근 출력: `ApproachResult`
```
motion: Motion           # approaching | receding | steady | unknown
```

## ④ STT 출력: `SpeechResult`  *(MVP 외 확장)*
```
text: str                # 인식된 문장 (없으면 "")
is_speech: bool          # 음성이 감지됐는지
confidence: float        # 0.0 ~ 1.0
lang: str | None         # 인식 언어 (예: "ko")

.to_korean() →  예: '"앞에 차가 지나갑니다"'
```

---

## 최종 통합: `FusedResult`
```
sound: ClassResult
direction: DirectionResult
approach: ApproachResult

.to_korean()  →  예: "사이렌, 후방, 접근 중"
```

---

## 모듈별 함수 시그니처 (약속)
| 모듈 | 함수 | 입력 → 출력 |
|------|------|------------|
| 분류 | `classifier.infer` | `AudioChunk → ClassResult` |
| 방향 | `doa.estimate_direction` | `AudioChunk → DirectionResult` |
| 접근 | `approach.ApproachDetector.update` | `AudioChunk → ApproachResult` |
| STT | `stt.Transcriber.transcribe` | `AudioChunk → SpeechResult` |

→ 각자 함수 **내부만** 구현하면 됨. 시그니처(입출력 형식)는 유지.
