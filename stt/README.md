# ④ STT(음성→텍스트)

주변의 확성기·외침·안내방송을 텍스트로 바꾼다. STT는 긴급음을 판정하지 않으며,
사이렌·경적 경보가 켜지면 분류 파이프라인이 STT를 중단한다.

## 설치와 실행

```bash
pip install -r stt/requirements.txt

python -m stt.run --wav speech.wav
python -m stt.run --mic
python -m stt.run --mic --respeaker

# 통합 파이프라인
python main.py --mic --channels 6 --stt
python main.py --mic --channels 6 --stt --stt-model small  # CPU 지연을 줄일 때
```

기본 설정은 다국어 `medium`, 한국어 고정, CPU `int8`/CUDA `float16` 자동 선택이다.
Mac CPU 검증에서 `medium`은 긴 문장 처리량은 실시간보다 빨랐지만 짧은 발화 한 건당
약 4.3초가 걸렸다. 빠른 자막이 더 중요하면 `small`을 사용하고 정확도를 다시 확인한다.

## 현재 처리 방식

1. 1초 입력을 Silero ONNX VAD로 검사한다.
2. 음성이 이어지는 동안 최대 8초까지 모은다.
3. 발화가 끝나면 faster-whisper가 한 번에 인식한다.
4. Whisper 내부 Silero VAD로 도로소음을 다시 거른다.
5. 휴리스틱 신뢰도가 0.4 미만인 자막은 표시하지 않는다.
6. 인식은 백그라운드 워커에서 수행한다. 긴급 진입 시 이전 세대의 대기·진행·완료
   자막을 모두 무효화한다.
7. 통합 대시보드는 완성 자막을 5초간 유지하고 새 자막으로 교체한다. 긴급 진입 시 즉시 지운다.

`vad_backend="auto"`의 폴백 순서는 Silero → WebRTC → energy다. `--vad N`을 지정하면
명시적으로 energy VAD로 바뀐다. RMS 정규화는 잡음도 키울 수 있어 기본값은 꺼져 있다.

## 운영 시 확인할 상태

- `last_error`: 인식 오류. 워커는 죽지 않고 다음 입력에서 복구를 시도한다.
- `dropped_chunks`: 디코딩이 입력보다 느려 큐가 찼을 때 생략한 1초 청크 수.
- `reset_count`: 긴급 경보 진입으로 STT가 초기화된 횟수.

콘솔과 대시보드는 오류 및 누적 드롭을 표시한다. 오류가 반복되거나 드롭이 계속 늘면
`--stt-model small`로 낮추거나 Jetson CUDA 경로를 사용한다.

## 검증

```bash
pytest -q tests/test_stt.py
python -m evaluation.download_figshare_samples --kind road --count 6
python -m evaluation.download_fleurs_samples --count 20
python -m evaluation.benchmark_stt --corpus fleurs --model medium
python -m evaluation.benchmark_stt --corpus tts --model medium
```

사람 음성 80건과 긴급차량 문장 TTS 40건의 결과 및 한계는
[STT 검증 문서](../docs/stt/validation.md)에 기록했다.

- 상세 설계: [design.md](../docs/stt/design.md)
- Jetson 배포: [jetson.md](../docs/stt/jetson.md)
