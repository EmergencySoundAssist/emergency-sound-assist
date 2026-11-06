# STT(음성→텍스트) 설계

목표는 평상시 주변 말을 `SpeechResult` 자막으로 바꾸는 것이다. 사이렌·경적 검출과
차종·방향·접근 판단에는 관여하지 않는다.

## 전체 흐름

```text
마이크 ch0 → 1초 묶음 → 외부 Silero VAD
                         ├─ 비음성: 버림
                         └─ 음성: 발화 버퍼(최대 8초)
                                      ↓ 발화 종료
                           faster-whisper medium
                           + 내부 Silero VAD
                           + 환각 임계값
                                      ↓
                         신뢰도 0.4 이상만 SpeechResult
```

Silero를 불러올 수 없을 때만 WebRTC, 그다음 energy VAD로 폴백한다. 외부 VAD는 발화의
시작과 끝을 정해 불필요한 디코딩을 줄이고, 내부 VAD와 신뢰도 필터는 외부 VAD를 통과한
도로소음의 오자막을 막는다.

## 긴급 경보와 STT의 관계

분류기가 사이렌·경적을 활성화하면 STT보다 긴급 경보를 우선한다.

- 평상시: 1초 오디오를 워커에 전달하고 완성된 자막을 가져온다.
- 자막 표시: 완성된 마지막 자막을 대시보드에서 5초 유지하고, 새 자막이 오면 즉시 교체한다.
- 긴급 진입: 한 번만 `reset()`하고 긴급 중에는 오디오를 전달하지 않는다.
- 긴급 유지: 매 tick reset하지 않는다.
- 긴급 해제: 새 세대의 음성부터 다시 수집한다.

`STTWorker`의 세대 번호는 reset 시 즉시 증가한다. 따라서 reset 전에 큐에 있던 입력,
이미 변환 중이던 결과, 아직 소비하지 않은 최신 자막은 모두 폐기된다. STT가 느려도
분류·방향·접근 루프는 백그라운드 인식 때문에 멈추지 않는다.

워커의 `latest()`는 완성 자막을 한 번만 전달한다. 콘솔은 이를 새 로그로 한 번 출력하고,
대시보드는 별도의 표시 캐시에 5초 보관한다. 사이렌·경적 진입 시 이 표시 캐시도 즉시 지워
오래된 평상시 자막이 긴급 화면 뒤에 다시 나타나지 않게 한다.

## 오류와 과부하 처리

- 엔진 예외는 워커를 종료시키지 않고 `last_error`로 공개한다.
- 다음 성공 시 오류 상태를 지운다.
- 큐가 가득 차면 오래 지연된 오디오를 쌓지 않고 새 청크를 생략하며
  `dropped_chunks`를 증가시킨다.
- 콘솔과 대시보드는 오류와 누적 드롭을 표시한다.

정확성보다 최신성이 중요한 실시간 보조장치이므로, 밀린 음성을 수십 초 뒤에 자막으로
내보내는 대신 드롭을 관측 가능하게 만든 선택이다.

## 주요 설정

| 설정 | 기본값 | 의미 |
|---|---:|---|
| `model_size` | `medium` | 한국어 정확도 우선. CPU 지연이 크면 `small` |
| `language` | `ko` | 한국어로 고정 |
| `device` / `compute_type` | `auto` | CPU/int8 또는 CUDA/float16 |
| `vad_backend` | `auto` | Silero → WebRTC → energy |
| `silero_threshold` | `0.5` | 외부 Silero 말소리 확률 임계값 |
| `silero_voiced_ratio` | `0.2` | 청크에서 요구하는 최소 말소리 비율 |
| `max_utterance_seconds` | `8.0` | 발화를 강제로 끊는 최대 길이 |
| `silence_release_chunks` | `1` | 음성 뒤 비음성 1청크에서 인식 시작 |
| `min_utterance_seconds` | `0.5` | 너무 짧은 입력 제거 |
| `whisper_vad_filter` | `True` | Whisper 내부 2차 Silero VAD |
| `min_confidence` | `0.4` | 저신뢰 자막 표시 억제 |
| `normalize_audio` | `False` | 잡음 증폭 위험 때문에 기본 OFF |

신뢰도는 `exp(avg_logprob)` 기반 휴리스틱이며 보정된 확률이 아니다. 임계값 0.4는 공개
도로소음 표본에서 환각을 제거하면서 평가 음성을 유지한 값으로, 실차 데이터가 생기면
반드시 다시 조정한다.

## 엔진 교체점

`Transcriber(engine=...)`가 받는 엔진은
`transcribe(samples, sample_rate) -> (text, confidence, language)`만 구현하면 된다.
테스트는 이 인터페이스의 가짜 엔진을 사용하므로 모델 설치 없이 상태·오류·reset 동작을
검증한다.

## 검증과 남은 한계

공개 FLEURS 한국어 사람 음성과 Figshare 도로소음을 섞은 실제 엔진 평가를 수행했다.
수치와 재현 명령은 [validation.md](validation.md)에 있다. 아직 검증되지 않은 부분은
실차 ReSpeaker 입력, 차량 스피커/확성기 음성, Jetson GPU 지연, 여러 사람이 겹쳐 말하는
환경이다.
