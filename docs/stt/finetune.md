# STT 파인튜닝 — 평가셋 v0 · baseline 측정 기록

설계: `docs/superpowers/specs/2026-07-06-stt-finetune-data-eval-design.md`
실행법: `finetune/README.md`

## 평가셋 v0 (SEED 20260706, 총 350발화)

| 구성 | 소스 | 수 |
|---|---|---|
| 일반 발화 | zeroth-korean test(CC-BY-4.0) 130 + google/fleurs ko_kr test(CC-BY-4.0) 70 | 200 |
| 긴급문구 | 75문장 × edge-tts 2화자(SunHi·InJoon) | 150 |

조건: corpus = clean/snr10/snr5/snr0 각 50. tts = loudspeaker 75(소음 SNR 10/5/0 순환)
+ 나머지 75 를 4조건 순환. 노이즈 = DEMAND TCAR/TBUS/STRAFFIC + MS-SNSD 차량 4종.

데이터 출처·라이선스: zeroth-korean(CC-BY-4.0), google/fleurs(CC-BY-4.0), DEMAND(Zenodo
1227121, CC BY-SA 3.0 으로 취급), MS-SNSD(코드 MIT, 오디오 CC0/CC BY-SA 혼재), edge-tts
합성음(Microsoft Edge 온라인 TTS 산출물). 평가셋을 팀원/Jetson 에 복사·배포할 때 출처 표기 필요.

## 스펙 편차 (2026-07-06 확정)

- Common Voice ko 제외 — Mozilla 가 2025-10 HF 배포 중단(Data Collective 이관).
- TTS 화자 풀 = edge-tts 한국어 3종뿐. 평가 SunHi+InJoon / 미래 학습 Hyunsu 로 분리.
- UrbanSound8K(6GB) 기본 제외 — 사이렌 겹침 조건 v0 미포함. `data/noise/extra/` drop-in 지원.
- 평가 엔진은 `stt.Transcriber` 재사용 대신 `WhisperModel` 직접 호출(STTConfig 파라미터
  미러링) — bias arm 인자(initial_prompt/hotwords) 때문. 런타임 코드는 불변.
- 숫자 표기 통일(10↔십) 미구현 — CER 페널티로 수용.
- manifest 는 스펙의 snr10/snr5/snr0 라벨 대신 `condition + snr_db` 인코딩 사용.

## 알려진 한계

- ReSpeaker ch0·실차 캐빈 음향 미반영(실차 녹음 가능 시 v1).
- TTS 절대 CER 은 편향 가능 — 긴급문구 지표는 arm 간 상대 비교용.
- 확성기는 근사 시뮬(밴드패스+클립+반사 1개), 실측 IR 아님.
- 숫자 표기(10↔십)는 CER 페널티로 남음.
- 발화 단위 파일 평가라 VAD/버퍼링 미경유(배포 경로와 차이).
- 참고용 WER 은 정규화 없이 원문 그대로 계산됨(스펙 §3의 정규화 규칙은 CER 에만 적용 —
  띄어쓰기 차이도 WER 오류로 집계).

## baseline 측정 결과 (2026-07-06 측정, n=350/arm)

| arm | CER(전체) | corpus clean | corpus noise@10 | corpus noise@5 | corpus noise@0 | 키워드 히트율 | 소요 | 비고 |
|---|---|---|---|---|---|---|---|---|
| small | 10.6% | 7.9% | 11.2% | 14.7% | 11.9% | 86.0% | 458s | |
| small-bias | 10.6% | 8.2% | 11.1% | 15.1% | 12.7% | 96.0% | 506s | |
| medium | 8.1% | 5.5% | 10.3% | 10.8% | 8.9% | 91.3% | 1240s | 배포 기본 |
| medium-bias | 7.9% | 6.5% | 9.4% | 10.4% | 9.1% | 99.3% | 1397s | |
| large-v3-turbo | | | | | | | | 보류 — Jetson CUDA 미감지 상태·오프라인이라 실기 측정 불가. docs/stt/jetson.md §3 GPU 설정 후 측정 예정 |

게이트 판단은 corpus(실음성) 컬럼 기준 — TTS 절대치는 상대 비교용.

확성기(loudspeaker, TTS 긴급문구) 세부: CER 은 낮지만(medium 3.1~3.2%, small 3.6~5.9%)
키워드 히트율은 전체 평균보다 눈에 띄게 낮다 — medium loudspeaker@0 84.0% / @5 88.0% / @10 88.0%
(vs medium 전체 91.3%), small loudspeaker@0 72.0% / @5 80.0% / @10 84.0% (vs small 전체 86.0%).
확성기 왜곡(밴드패스+클립)이 짧은 명령형 문구의 핵심 동사를 삼키는 경향 — bias(hotwords/initial_prompt)
로 크게 개선(medium-bias loudspeaker@0 96.0%, small-bias loudspeaker@0 88.0%).

## 오답 분석 (medium arm, ref 3자 이상, per-item CER 상위)

- **표기 불일치(숫자)**: 최대 오답 다수가 실제 인식 실패가 아니라 참조문의 한글 숫자(마흔 일곱, 이천 오 년)
  대 모델 출력의 아라비아 숫자(47, 2005) 불일치로 인한 CER 페널티(known limitation, 위 "알려진 한계" 참고).
  예: `노르웨이는 육아휴직 마흔 일곱 주를...` → `...47주를...` (CER 32.0%).
- **확성기·소음 조건에서 긴급 단문 오인식**: 짧은 명령형 문구가 왜곡 하에서 다른 단어로 치환됨
  — `뒤에요`→`G.A.O.`(CER 100%, loudspeaker@0), `짐 떨어져요`→`지금 떨어져요`(40.0%, loudspeaker@5),
  `불났어요`→`불렀어요`/`불냈어요`(25.0%, loudspeaker@0/noise@0). 긴급문구 키워드 실패의 핵심 원인.
- **긴 일반 문장(zeroth/fleurs)의 저 SNR 열화 및 드문 환각**: noise@5/@0 에서 어순·조사 오류가 누적되고,
  한 건은 무관한 문장이 덧붙는 환각(noise@5, zeroth, `...또 말씀하셔가지고 아직 잊지 않으세요...`)이 관측됨.
  로마자 병기 고유명사(schengen zone, cristina fernandez...)도 fleurs 소스에서 반복적으로 CER 을 키움.

## Go/No-Go 판단

기준(스펙 §4 초안): ① corpus 기준 SNR≤5dB CER 이 corpus/clean 대비 2배↑ 열화, ② 긴급문구 키워드
히트율 <90% — 단, hotwords(bias) arm 만으로 기준을 충족하면 파인튜닝 보류. 배포 기본값인 **medium**
을 판단 기준으로 삼고(small 은 참고), 아래는 corpus/noise@5 대 corpus/clean 비율로 계산.

| arm | 기준① corpus noise@5 / clean | 배율 | 기준① 게이트 걸림(≥2배)? | 기준② 키워드 히트율 <90%? |
|---|---|---|---|---|
| small | 14.7% / 7.9% | 1.85배 | 아니오 | 예 — 86.0% (게이트 걸림) |
| small-bias | 15.1% / 8.2% | 1.84배 | 아니오 | 아니오 — 96.0% |
| medium | 10.8% / 5.5% | 1.98배 (임계 근접) | 아니오 | 아니오 — 91.3% (기준선 90% 상회하나 여유 1.3%p 뿐) |
| medium-bias | 10.4% / 6.5% | 1.61배 | 아니오 | 아니오 — 99.3% |

정리: **medium(배포 기본) 은 기준①·② 모두 형식상 미달(게이트가 걸리지 않음)** — 그러나 CER 배율
1.98배는 2배 문턱에 사실상 근접해 있고, 키워드 히트율 91.3%도 90% 기준선 위 1.3%p 여유밖에 없어
경계선 통과에 가깝다. 게다가 확성기 조건만 떼어 보면 키워드 히트율이 84~88%로 90% 밑으로 떨어지는
하위 조건이 존재한다(전체 평균에는 가려짐). medium-bias(hotwords) arm 은 두 기준 모두 넉넉한
여유로 충족(1.61배, 99.3%)하여, "hotwords arm 만으로 기준 충족 시 파인튜닝 보류" 조항에 해당한다.

- **판단(권고): 파인튜닝 보류를 권고.** medium-bias(hotwords/initial_prompt) 조합이 게이트 기준
  ①②를 모두 충분한 여유로 충족하고, 확성기 조건의 키워드 히트율 약점도 bias 로 크게 개선된다
  (loudspeaker@0 84.0%→96.0%). 파인튜닝(학습셋 구축) 투자보다 hotwords/initial_prompt 튜닝을
  배포 기본으로 채택하는 편이 비용 대비 이득이 크다고 판단된다.
- **근거**: ① CER 배율 — medium 1.98배(임계 근접·미달), medium-bias 1.61배(충족 여유 있음).
  ② 키워드 히트율 — medium 91.3%(bare 기준선 위·확성기 조건 84~88%로 하위 조건 존재),
  medium-bias 99.3%(전 조건 96%+). 확성기 왜곡이 긴급 단문의 핵심 동사를 삼키는 실패 모드가
  오답 분석에서 확인되었고, bias 가 이를 직접적으로 완화한다.
- **단서**: 이 판단은 v0 평가셋(edge-tts 시뮬레이션 확성기·근사 노이즈, ReSpeaker/실차 캐빈 미반영)
  기준이며, large-v3-turbo 는 미측정(보류)이다. **최종 결정(파인튜닝 착수 여부)은 사용자 몫이다.**
