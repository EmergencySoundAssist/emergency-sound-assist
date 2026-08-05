# STT 공개 데이터 검증

## 결론

현재 기본 구조는 `외부 Silero VAD → Whisper 내부 Silero VAD → 신뢰도 필터`다.
공개 도로소음 표본에서는 WebRTC 단독보다 발화 종료와 오자막 방지가 모두 좋아졌으므로
이 구조를 유지한다. 다만 아래 수치는 실차 정확도가 아니라 공개 데이터 기반 사전 검증이다.

## 환경과 데이터

- 엔진: faster-whisper 1.2.1, 다국어 `medium`, CPU/int8, beam 1
- 실행 환경: Apple Silicon Mac CPU, Python 가상환경
- 사람 음성: Google FLEURS `ko_kr` dev 중 8초 이하 20문장
- 도로소음: Figshare Environmental Sound Classification 50의 `Road Noises` 6개
- 혼합 조건: clean, 20 dB, 10 dB, 5 dB SNR
- 제품 경로: 1초 입력, 발화 VAD와 버퍼, 내부 VAD, 신뢰도 필터를 모두 통과

FLEURS는 실제 사람이 녹음했지만 차량 안 녹음은 아니다. 도로소음도 나중에 합성한 것이므로
실차 마이크·잔향·AEC·스피커 왜곡까지 재현하지 않는다.

## VAD 구조 비교

동일한 `small` 모델, 한국어 TTS 5문장, clean/10 dB 조건과 도로소음 6개로 비교했다.

| 구조 | 소음 엔진 호출 | 소음 오자막 | 10 dB 발화 종료 | 10 dB CER |
|---|---:|---:|---:|---:|
| WebRTC 외부 VAD만 | 6/6 | 2/6 | 0% | 10.7% |
| WebRTC + 내부 Silero + 신뢰도 | 6/6 | 0/6 | 0% | 10.7% |
| 외부 Silero + 내부 Silero + 신뢰도 | 1/6 | 0/6 | 100% | 7.4% |

WebRTC의 aggressiveness 0~3과 voiced ratio 0.2~0.7을 전수 비교했지만 이 도로소음 6개는
모든 조합에서 한 번 이상 음성으로 판정됐다. 따라서 WebRTC 임계 조절이 아니라 Silero를
기본으로 바꿨다.

## FLEURS 사람 음성 결과

각 SNR 20건, 총 80건이다. 한국어는 띄어쓰기 변형이 커서 CER를 주 지표로 본다.

| SNR | 평균 CER | 정확 일치 | 발화 종료 | 평균 변환시간 | 실시간 계수 |
|---|---:|---:|---:|---:|---:|
| clean | 10.2% | 35% | 100% | 4.30초 | 0.63 |
| 20 dB | 11.6% | 35% | 100% | 4.30초 | 0.63 |
| 10 dB | 12.3% | 35% | 100% | 4.32초 | 0.64 |
| 5 dB | 19.2% | 25% | 100% | 4.51초 | 0.67 |

도로소음만 8초씩 넣은 6건 중 외부 VAD가 엔진을 호출한 것은 1건이었고, 최종 오자막은
0건이었다. 깨끗한 원본 CER가 이미 10.2%이므로 20/10 dB의 추가 손실은 비교적 작고,
5 dB에서 손실이 뚜렷해진다.

## 긴급차량 문장 TTS 결과

구급차·경찰차·소방차·접근·정지 등을 포함한 10문장에 같은 네 조건을 적용했다.

| SNR | 평균 CER | 정확 일치 | 발화 종료 |
|---|---:|---:|---:|
| clean | 2.4% | 80% | 100% |
| 20 dB | 0.9% | 90% | 100% |
| 10 dB | 3.6% | 70% | 100% |
| 5 dB | 3.8% | 80% | 100% |

짧은 TTS는 평균 오디오가 약 2.4초인데 변환은 평균 4.3초여서 실시간 계수가 약 1.78이다.
백그라운드 워커 덕분에 긴급음 검출은 멈추지 않지만, CPU 자막은 늦게 표시될 수 있다.
또한 강한 소음에서 `구급차→고급차/도둑차`, `소방차→수강차`처럼 핵심 단어 오류가
실제로 발생했으므로 자막을 안전 판단의 단독 근거로 쓰면 안 된다.

## 재현

```bash
pip install -r stt/requirements.txt
python -m evaluation.download_figshare_samples --kind road --count 6
python -m evaluation.download_fleurs_samples --count 20

python -m evaluation.benchmark_stt \
  --corpus fleurs --model medium --snr clean,20,10,5 \
  --output data/stt_evaluation_fleurs_medium.json

python -m evaluation.benchmark_stt \
  --corpus tts --model medium --snr clean,20,10,5 \
  --output data/stt_evaluation_tts_medium.json
```

평가 결과와 내려받은 데이터는 `data/` 아래에 저장되며 Git에는 포함하지 않는다.

## 다음 검증

1. ReSpeaker ch0로 차량 내부에서 사람 음성·확성기·라디오·창문 개폐를 녹음한다.
2. 거리와 차량 속도별로 음성 누락률, CER, 오자막/시간, 지연을 측정한다.
3. Jetson GPU에서 모델별 지연과 워커 `dropped_chunks`를 비교한다.
4. 그 결과로 `medium`/`small`, VAD 임계값, 신뢰도 0.4를 최종 보정한다.
