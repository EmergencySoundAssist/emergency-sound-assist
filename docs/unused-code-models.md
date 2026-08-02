# 사용 여부와 정리 결과

현재 저장소의 `main.py → pipeline.runner.Pipeline` 통합 경로와 문서에 적힌 독립 실행 명령을
기준으로 코드·모델을 점검했다. 마지막 점검일은 2026-08-02다.

## 현재 모델

`classifier/models/`에는 현재 사용하는 모델만 남겼다. 네 모델 모두 ONNX Runtime CPU
세션으로 정상 로드되는 것을 확인했다.

| 모델 | 현재 사용 위치 | 실행 조건 | 판단 |
|---|---|---|---|
| `cnn_attn_full_s42.onnx` + `.data` | `classifier.inference._session()` | 통합·분류 실행의 5초 확정 검출 | 필수 |
| `cnn_attn_full_s42_87f.onnx` | `_fast_session()` | 약 2초 PRE 예비경보 | 사용 |
| `subtype_cnn_attn_yt_s42.onnx` | `_subtype_session()` | 사이렌 차종 분류·투표 | 사용 |
| `speed_neural_dir.onnx` | `_speed_session()` | 사이렌 확정 및 실제 5초 창 준비 후 | 사용 |

네 모델이 시작과 동시에 모두 로드되는 것은 아니다. 5초 검출은 필수이고, PRE·차종·접근
모델은 해당 기능이 필요할 때 지연 로드된다.

`.gitignore`에는 이 네 모델을 명시적으로 허용하는 예외를 추가했다. 따라서 이전에 누락됐던
PRE·차종·접근 모델도 다음 Git 커밋에 포함할 수 있다. 모델을 별도 서버에서 내려받는 구조가
아니므로 프로젝트 변경을 커밋할 때 세 파일도 함께 추가해야 한다.

### 실제 모델 출력 사용

| 모델 | 출력 | 제품에서 쓰는 값 |
|---|---|---|
| 5초 검출 | `logits[3]` | 사이렌·경적·일반 소리 확정 |
| 2초 검출 | `logits[3]` | PRE 예비경보 |
| 차종 | `logits[3]` | 구급차·경찰차·소방차 확률 |
| `speed_neural_dir` | `speed[1]`, `f0[216]`, `dir[3]` | `dir[3]`은 조건부 융합, `speed[1]`은 진단만, `f0`는 현재 미사용 |

STT의 Whisper 모델과 Silero VAD 모델은 이 폴더에 포함되지 않는다. `--stt`를 켰을 때
faster-whisper/CTranslate2와 설치 패키지의 캐시를 통해 별도로 사용한다.

## 이번에 제거한 항목

현재 저장소 안에서 호출되지 않고 현재 파이프라인으로 대체된 항목만 제거했다.

| 파일 | 제거한 항목 | 이유 |
|---|---|---|
| `classifier/models/` | `speed_neural.onnx`, `speed_neural.onnx.data` | `speed_neural_dir`로 대체됐고 로더 참조가 없음 |
| `classifier/inference.py` | `SPEED_ENABLED`, `_kmh_to_level()`, `_infer_speed()`, `speed_dir()` | 과거 속도 단계·단순 방향 반환 경로. 현재는 `speed_evidence()` 사용 |
| `pipeline/alert.py` | `SPEED_TIERS`, `speed_tier()`, `DIR_KO`, `dir_tier()`, `SpeedTracker` | 현재 이동 판단은 `ConditionalMotionFusion`이 담당 |
| `core/types.py` | `FusedResult` | 생성·반환 코드가 없고 실제 통합 출력은 `AlertEvent + info` |
| `core/types.py` | `CHANNELS = 4` | 참조가 없고 물리 마이크 4개와 USB 6채널을 혼동시킴 |
| `core.types.ClassResult` | 미사용 `speed_level`, `speed_kmh` 필드 | 속도 모델 값은 통합 진단 `info`에서만 관리 |
| `pipeline/runner.py`·`alert.py` | 항상 `None`이던 `info.speed_level`과 화면 표시 분기 | 제품 속도 단계는 표시하지 않기로 확정 |

구형 모델 두 파일은 Git 이력에서 복구할 수 있다. 활성 `speed_neural_dir.onnx`는 삭제하지
않았고 현재 조건부 이동 융합에 계속 사용한다.

분류 단독 API에서 사용되지 않던 `_infer_subtype()`은 삭제하지 않고 `classifier.infer()`에
연결했다. 따라서 `classify.py`에서도 사이렌이면 차종 결과를 받을 수 있다. 통합 실행은 기존처럼
여러 tick의 `subtype_probs()`를 투표한다.

## 통합 경로 밖에서 유지한 코드

다음은 `main.py`가 직접 호출하지 않아도 독립 진단·검증에 필요하므로 보존했다.

| 코드 | 용도 |
|---|---|
| `classify.py`, `classifier.infer()` | 분류·차종 단독 실행 |
| `doa/live.py`, `doa/estimator.py` | ReSpeaker 자체 DoA 레지스터 브링업 |
| `doa/multi_live.py`, `doa/tracking.py`, `doa/led_ring.py` | 다중 채널 방향 단독 실행·스무딩·LED |
| `doa/diag.py`, `doa/respeaker_tuning.py` | 장착 각도와 USB 제어 진단 |
| `evaluation/` | 공개 데이터 평가와 결과 재현 |
| `tests/` | 회귀 테스트 |

## 제품 범위 결정 후 판단할 항목

| 파일/기능 | 현재 상태 | 판단 기준 |
|---|---|---|
| `approach/test_detector.py` | 합성 통과 교육·단독 실행 스크립트 | `evaluation/`만 사용할 것이 확정되면 이동 또는 제거 가능 |
| `pipeline.alert.GpioSink` | `make_sink("gpio")`로 생성 가능하지만 `main.py` CLI에는 아직 미연결 | 진동 모터를 쓰면 CLI·하드웨어 테스트 추가, 쓰지 않으면 제거 |
| `doa.live` 자체 DoA 경로 | 통합은 원시 `ch1~4` SRP-PHAT 사용 | 저사양 폴백·장치 진단 용도로 유지 권장 |

이 세 항목은 현재 미사용이라고 단정할 수 없거나 하드웨어 계획에 달려 있어 임의로 삭제하지
않았다.
