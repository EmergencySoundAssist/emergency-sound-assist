# 소리 분류 설계

현재 저장소는 학습이 아니라 배포용 ONNX 추론을 담당한다. 초기 모델 비교·학습 계획은
현재 런타임 범위가 아니므로 이 문서에서 다루지 않는다.

## 모델 구성

| 파일 | 입력 창 | 역할 | 현재 사용 |
|---|---:|---|:---:|
| `cnn_attn_full_s42_87f.onnx` | 약 2초, 87 frame | 사이렌 PRE 예비검출 | 사용 |
| `cnn_attn_full_s42.onnx` + `.data` | 5초, 216 frame | siren/horn/noise 확정 검출 | 사용 |
| `subtype_cnn_attn_yt_s42.onnx` | 5초 | 구급차/경찰차/소방차 | 사용 |
| `speed_neural_dir.onnx` | 실제 5초 | 정지/접근/멀어짐 증거 | 사용 |

모든 활성 ONNX 모델은 `classifier/inference.py`가 lazy load한다. provider 우선순위는
TensorRT, CUDA, CPU다.

## 전처리

```text
16 kHz ch0
  → 22.05 kHz 리샘플
  → 5초 롤링 버퍼
  → log-mel (n_fft=1024, hop=512, n_mels=64)
  → 창별 평균/표준편차 정규화
  → (1, 1, 64, 216)
```

PRE 모델은 같은 멜의 끝 87 frame을 별도로 정규화해 사용한다. 차종과 움직임 모델은 확정
검출과 같은 5초 멜을 재사용하므로 추가 STFT를 만들지 않는다.

## 검출과 경보 상태기계

모델 softmax만으로 경보를 바로 켜지 않고, 목표 클래스 logit과 나머지 최대 logit의 차이인
마진을 `pipeline.alert.Gate`에 전달한다.

```text
2초 마진 → 빠른 PRE 상태기계
5초 siren 마진 → 사이렌 확정 상태기계
5초 horn 마진 → 경적 상태기계
```

상태기계는 연속 조건, K/M 투표, 켜기/끄기 히스테리시스, hangover, 리마인더를 사용해
단일 tick의 튐을 경보로 바로 표시하지 않는다.

## 사이렌 확정 뒤 처리

- 차종 확률을 경보 중 약 6초 다수결해 구급차/경찰차/소방차를 표시한다.
- `speed_neural_dir`는 무음 패딩이 없는 실제 5초가 쌓인 뒤에만 실행한다.
- 움직임 dir head는 음량·직접 도플러와 조건부 융합한다.
- speed head 원시값은 디버그에만 남기고 km/h나 속도 단계로 표시하지 않는다.

## 독립 API와 통합 API

- `classifier.infer(chunk) → ClassResult`: 분류 단독 실행용
- `classifier.analyze(chunk) → dict | None`: PRE/확정 마진을 포함한 통합용
- `classifier.subtype_probs()`: 마지막 5초 창의 차종 확률
- `classifier.speed_evidence()`: 마지막 실제 5초 창의 움직임 모델 증거

실행 명령은 [공통 실행 문서](../running.md#4-소리-분류만-실행)에 있다.

## 한계

- 시작 직후 5초 창은 무음 패딩을 포함한다. 움직임 모델은 이 구간에서 실행하지 않지만 검출
  모델의 워밍업 출력은 안정 구간과 다를 수 있다.
- 차종과 검출의 실제 도로 source-disjoint 평가는 추가로 필요하다.
- TensorRT provider가 없는 환경에서는 CUDA 또는 CPU로 자동 폴백한다.
- ONNX 학습·변환 원본은 별도 학습 저장소에 있으며 이 저장소는 재학습 코드를 포함하지 않는다.
