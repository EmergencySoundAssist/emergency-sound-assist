# EmergencySoundAssist

청각장애 운전자를 위해 주변의 긴급 소리와 음성을 분석해 **소리 종류·차종·방향·이동 상태·자막**을
제공하는 프로젝트다.

현재 통합 파이프라인은 다음 기능을 연결한다.

| 기능 | 현재 구현 |
|---|---|
| 긴급음 검출 | 2초 PRE 예비경보, 5초 사이렌·경적 확정 |
| 사이렌 차종 | 구급차·경찰차·소방차 분류 |
| 방향 | ReSpeaker 원시 4채널의 SRP-PHAT/MUSIC 또는 장치 자체 DoA |
| 접근·멀어짐 | `speed_neural_dir` 모델·음량 변화·직접 도플러의 조건부 융합 |
| 평상시 음성 | faster-whisper 기반 한국어 STT, 긴급음 중 자동 중단 |

최종 화면에는 예를 들어 `구급차 · 후방 · 접근 중`과 같은 경보가 표시된다. 접근 모델의
원시 `speed` 값은 km/h로 검증되지 않았으므로 현재 사용자 화면에는 속도로 표시하지 않는다.

## 빠른 실행

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt

python main.py --demo
python main.py --mic --channels 6 --view dashboard
python -m pytest -q
```

STT를 사용할 때는 STT 의존성을 추가한 뒤 실행한다.

```bash
pip install -r stt/requirements.txt
python main.py --mic --channels 6 --stt --stt-model small
```

전체 실행 옵션, 단독 모듈 실행, 평가 재현, Jetson GPU·TensorRT 확인 방법은
**[실행 명령어](docs/running.md)**에 한곳으로 정리했다. Jetson에서는 일반 PyPI 패키지가
CUDA/TensorRT 빌드를 덮어쓸 수 있으므로 해당 문서의 Jetson 절을 먼저 확인한다.

## ReSpeaker 채널 사용

ReSpeaker는 물리 마이크가 4개지만 USB 입력은 6채널이다.

| USB 채널 | 용도 |
|---|---|
| `ch0` | 분류·차종·접근·STT |
| `ch1~4` | 방향 추정용 원시 마이크 4채널 |
| `ch5` | 재생 참조, 현재 미사용 |

## 저장소 구조

```text
audio/        마이크·WAV 입력
classifier/   긴급음 검출과 사이렌 차종 분류
doa/          방향 추정·다중 음원·LED·진단
approach/     접근 모델과 신호 특징 계산
stt/          VAD·faster-whisper 자막
pipeline/     경보 상태기계와 조건부 이동 융합
core/         공통 타입
evaluation/   접근·STT 공개 데이터 평가
tests/        단위·통합 테스트
docs/         설계·실행·검증 문서
```

## 문서

- [문서 전체 목차](docs/README.md)
- [실행 명령어](docs/running.md)
- [전체 구조와 동작 흐름](docs/architecture.md)
- [하드웨어와 채널 구성](docs/hardware.md)
- [데이터 인터페이스](docs/interfaces.md)
- [사용 여부와 정리 후보](docs/unused-code-models.md)
- [접근·멀어짐 설계](docs/approach/design.md) · [검증 결과](docs/approach/validation.md)
- [STT 설계](docs/stt/design.md) · [검증 결과](docs/stt/validation.md)
