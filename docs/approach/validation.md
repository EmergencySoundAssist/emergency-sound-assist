# 접근·멀어짐/속도 융합 검증

검증일: 2026-08-02

## 결론

- 제품에는 **보수적인 조건부 C 융합**을 채택했다. 다만 이것이 제약 없는 절대 최선으로
  입증된 것은 아니다.
- `speed_neural_dir.onnx`, 음량 추세, 직접 도플러를 모두 계산하지만 고정 평균하지 않는다.
- 직접 도플러는 사이렌 자체 주파수 변조에 취약하므로 단독 판정기가 아니라, 모델과 같은
  방향일 때만 음량 판단을 뒤집을 수 있는 충돌 보조 근거다.
- 차량 속도 3단계는 검증 성능이 부족하다. 모델 원시 km/h와 음량 변화 단계는 진단값으로만
  남기고, 제품 화면·위험도에는 속도로 표시하지 않는다.

## 데이터와 시나리오

- 공개 원본: Figshare의 *Large-Scale Audio Dataset for Emergency Vehicle Sirens and Road Noises*
  중 구급차 사이렌 WAV 10개. 라이선스 CC0.
- 이동 합성: `pyroadacoustics`로 도플러, 거리 감쇠, 노면 반사, 배경 잡음을 적용.
- 원본마다 6개 시나리오, 총 60개:
  - 정적: 음량 일정 / 증가 / 감소
  - 통과: 20 / 40 / 60 km/h, 측면거리 8 m
- 모델 비교는 실제 신호가 5초 쌓인 뒤부터 0.5초 간격으로 수행했다. 무음 패딩된 5초 미만
  입력은 `speed_neural_dir`가 이동을 오판해 제외했다.
- C의 문턱은 원본 파일 단위 leave-one-source-out 교차검증으로 선택했다. 같은 원본에서 만든
  시나리오가 학습·검증 fold에 동시에 들어가지 않게 분리했다.

원본: [Figshare dataset](https://figshare.com/articles/media/Large-Scale_Audio_Dataset_for_Emergency_Vehicle_Sirens_and_Road_Noises/19291472)
시뮬레이터: [pyroadacoustics](https://github.com/steDamiano/pyroadacoustics)

## 접근·멀어짐 결과

안전 점수는 접근 재현율에 50%, 멀어짐과 유지 재현율에 각각 25%를 둔 비교 지표다.

| 방식 | 안전 점수 | 균형 정확도 | 접근 재현율 | 멀어짐 재현율 | 유지 재현율 |
|---|---:|---:|---:|---:|---:|
| 조건부 C | **0.743** | **0.692** | 0.894 | **0.665** | 0.518 |
| 음량(deadband 0.20) | 0.741 | 0.690 | 0.894 | 0.656 | 0.518 |
| B(기존 문턱) | 0.596 | 0.585 | 0.628 | 0.248 | 0.879 |
| 모델 단독 | 0.450 | 0.489 | 0.333 | 0.163 | 0.972 |
| 직접 도플러 단독 | 0.322 | 0.296 | 0.400 | 0.487 | 0.000 |

공통 문턱을 전체 60개 시나리오에 적용하면 C가 1위지만 음량보다 안전 점수가 0.002 높을
뿐이다. 차이는 매우 작아서 C가 통계적으로 우월하다고 말할 수는 없다.

각 방식의 문턱을 똑같이 source-wise 튜닝해 held-out 원본에 적용한 더 엄격한 비교에서도
C 0.734, 음량 0.733, 튜닝 B 0.731이었다. C가 1위이기는 하지만 차이가 0.1~0.3%p라서
세 방식은 사실상 동률로 보는 편이 정직하다. 프로젝트 요구사항인 “전달받은 접근 모델과 직접
도플러를 모두 사용”도 만족하고 성능 손실이 없는 C를 최종 선택했다.

10개 source-wise fold 중 8개가 같은 C 조건을 선택했다.

- 음량 유지 deadband: `abs(log-power slope) <= 0.20`
- 모델 방향 신뢰도: 0.55 이상
- 직접 도플러 유효도: 0.20 이상

원시 speed head 값은 차량 km/h로 검증되지 않아 융합 판단에도 쓰지 않고 진단 로그에만 남긴다.

## 속도 단계 결과

접근 정답 구간 180 tick은 20/40/60 km/h가 각각 60개로 균형이다.

| 3단계 방식 | 정확도 | 출력률 |
|---|---:|---:|
| 모델 단독 | 33.9% | 100% |
| 음량 단독 | 42.2% | 92.2% |
| 모델·음량 단계가 같을 때만 출력 | 47.9% | 40.6% |

둘로 단순화해도 충분하지 않았다.

- 20 vs 40·60: 모델 68.3%, 다수 클래스 기준선 66.7%
- 20·40 vs 60: 가장 좋은 결과도 65.6%, 다수 클래스 기준선 66.7% 이하

원본 사이렌별 음압·녹음 상태 차이가 속도 차이보다 크게 나타났다. 따라서 지금 수치로
`느림/보통/빠름`을 표시하면 그럴듯하지만 신뢰할 수 없는 결과가 된다.

## 제품 규칙

1. 사이렌 확정 중 최근 3초에서 음량 기울기와 직접 도플러 진단을 계산한다.
2. `speed_neural_dir`는 실제 5초 오디오가 준비된 뒤에만 실행한다.
3. 모델과 음량이 같은 방향이면 그대로 채택한다.
4. 충돌하면 음량을 기본값으로 둔다. 신뢰도 높은 모델과 유효한 직접 도플러가 **동시에 같은
   방향**일 때만 모델 쪽으로 뒤집는다.
5. 음량 판단이 아직 불가능하면 신뢰도 높은 모델로 임시 폴백하고, 둘 다 불충분하면 기권한다.
6. 약 1.5초 다수결로 표시 깜빡임을 줄인다.
7. 속도 값은 디버그 정보에만 보존하고 경보 문구에는 넣지 않는다.

구현: `pipeline/motion_fusion.py`, `pipeline/runner.py`

## 재현

```bash
source .venv/bin/activate
pip install -r evaluation/requirements.txt
python -m evaluation.download_figshare_samples --count 12
python -m evaluation.benchmark_motion --sources 10 \
  --output data/evaluation_results_10sources.json
python -m evaluation.tune_fusion data/evaluation_results_10sources.json
python -m evaluation.compare_methods_cv data/evaluation_results_10sources.json
python -m evaluation.evaluate_speed_output data/evaluation_results_10sources.json
```

## 한계와 다음 검증

- 공개 원본은 이번 평가에서 구급차 사이렌만 사용했다. 경찰차·소방차 원본도 추가해야 한다.
- 시뮬레이션은 물리적 도플러와 감쇠를 포함하지만 실차의 차체 차음, 반사, 바람, 교통 혼합음,
  ReSpeaker 채널 특성을 완전히 재현하지 못한다.
- 절대/단계 속도를 다시 표시하려면 실제 차량 속도 라벨이 있는 여러 사이렌 소스와 장소로
  source-disjoint 검증을 하고, 최소한 다수 기준선을 의미 있게 넘어야 한다.
