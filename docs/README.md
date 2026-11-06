# 프로젝트 문서

처음 실행하려면 [실행 명령어](running.md), 전체 동작을 이해하려면
[전체 구조](architecture.md)부터 읽는다.

## 시작하기

- [실행 명령어](running.md) — 설치, 통합 실행, 모듈별 실행, 테스트, GPU 확인
- [전체 구조와 파이프라인](architecture.md)
- [하드웨어와 ReSpeaker 6채널](hardware.md)
- [데이터 인터페이스](interfaces.md)
- [용어집](glossary.md)
- [사용 여부와 정리 후보](unused-code-models.md) — 모델·레거시 코드·독립 도구 구분

## ① 소리 분류와 차종

- [소리 분류 설계](classifier/design.md)
- [구급차·경찰차·소방차 분류](classifier/subtype.md)

## ② 방향 추정

- [방향 추정 설계](doa/design.md)
- [실행 옵션](doa/running.md)
- [각도와 차량 4방향 매핑](doa/direction-mapping.md)
- [다중 음원 SRP-PHAT/MUSIC](doa/multi-source.md)
- [Jetson 배포](doa/jetson.md)

## ③ 접근·멀어짐

- [조건부 융합 설계](approach/design.md)
- [공개 데이터 검증](approach/validation.md)

## ④ STT

- [STT 설계](stt/design.md)
- [공개 데이터 검증](stt/validation.md)
- [Jetson 배포와 GPU 가속](stt/jetson.md)

## 문서 기준

- 현재 실행 동작의 기준은 `main.py`, `pipeline/runner.py`와 각 모듈 코드다.
- 학습 전용 저장소의 초기 실험 계획은 이 문서에서 제거하고 현재 포함된 ONNX 배포물만 설명한다.
- 공개 데이터 검증과 실제 차량 검증을 구분한다. 실차에서 확인하지 않은 수치는 제품 정확도로
  표현하지 않는다.
- 실행 명령은 저장소 루트와 `.venv` 활성화를 기준으로 한다.
