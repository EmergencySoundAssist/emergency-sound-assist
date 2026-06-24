# DoA 실행 명령어 모음

방향추정(DoA) 코드를 돌릴 때 쓰는 명령어 정리. **항상 `python` 사용**(이 프로젝트 conda 환경 기준 — `python3`는 시스템 파이썬을 가리킬 수 있음).

> **설정은 [`doa/config.py`](../../doa/config.py) 한 곳에서 관리.** 매번 플래그를 치기 싫으면
> config.py 의 값(hop/hold/algo/num/led + 보정값)을 바꾸면 된다. 그래도 아래 CLI 플래그를
> 주면 그 실행에 한해 덮어쓴다. (예: config 에서 `LED=True` 로 두면 `--led` 없이도 켜짐)

---

## 0. 환경 (최초 1회 + 작업 시작 시)
```bash
# 최초 1회
conda create -n airacle python=3.10 -y
conda activate airacle
pip install -r requirements.txt

# 작업 시작할 때마다
conda activate airacle          # 프롬프트가 (airacle) 인지 확인
```

## 1. 테스트 (하드웨어 불필요)
```bash
python -m pytest tests/ -q                    # 전체
python -m pytest tests/test_multi_source.py   # 다중방향 peak 로직
python -m pytest tests/test_led_ring.py       # LED 매핑/payload
```

## 2. 단일 방향 (1단계 — ReSpeaker 자체 DoA)
```bash
python -m doa.estimator        # 각도→4방향 매핑 데모 (장치 있으면 실측 1회)
```

## 3. 다중 방향 — 1회 캡처
```bash
python -m doa.multi_source     # 1초 캡처 → 동시 여러 방향 출력
```

## 4. 다중 방향 — 실시간 (주력)
```bash
python -m doa.multi_live                              # 기본
python -m doa.multi_live --led                        # + LED 링 점등
python -m doa.multi_live --led --algo music --num 3   # MUSIC, 최대 3개
```
- **2개 이상 감지되면** `#1 ★다중(2) ...` 처럼 영구 로그로 쌓이고, 단일/조용함은 한 줄 실시간 갱신.

### 옵션
| 옵션 | 기본 | 의미 |
|---|---|---|
| `--led` | off | 감지 방향을 LED 링에 점등 |
| `--algo` | SRP | `SRP`(가벼움) / `MUSIC`(다중 분리 유리, 무거움) |
| `--num` | 2 | 최대 동시 음원 수 (4-mic는 최대 3) |
| `--window` | 0.4 | 분석 창(초) — 길수록 방향 해상도↑ |
| `--hop` | 0.1 | 갱신 간격(초) — 짧을수록 빠름(≈10Hz). `⚠느림` 뜨면 늘릴 것 |
| `--height` | 0.5 | 2번째 peak 임계(최대 대비). 낮추면 약한 음원도 잡음 |
| `--min-sep` | 30 | 두 음원 최소 각도 간격(도). 낮추면 가까운 음원 분리 |
| `--threshold` | 0.01 | 이 RMS 미만이면 '조용함' |

### 자주 쓰는 조합
```bash
python -m doa.multi_live --led --hop 0.05 --window 0.3       # 더 빠른 반응(≈20Hz)
python -m doa.multi_live --algo music --num 3 --height 0.2   # 3개 동시 감지 도전
python -m doa.multi_live --height 0.25 --min-sep 20          # 약하고 가까운 음원까지
```

## 5. LED 단독 점검 (오디오 무관)
```bash
python -m doa.led_ring         # LED 한 칸씩 회전 — 링 동작 확인
```

---

## 트러블슈팅
| 증상 | 해결 |
|---|---|
| `No module named ...` | `conda activate airacle` 했는지, `python`(not `python3`) 쓰는지 확인 |
| `Invalid number of channels` | 기본 입력이 ReSpeaker가 아님 → 코드가 자동 탐지하지만, `sd.query_devices()`로 6채널 장치 확인 |
| LED `Access denied` | `sudo python -m doa.led_ring` 또는 udev 규칙(리눅스/Jetson) |
| `⚠느림` 표시 | `--window` 줄이기 / `--algo srp` / `--hop` 늘리기 |
| 방향은 맞는데 LED 위치가 회전됨 | `doa/led_ring.py` 의 `LED_OFFSET` 조정(한 칸=30°) |
| 좌/우가 반대 | `doa/estimator.py` 의 `MIRROR` 토글 (docs/doa/direction-mapping.md) |
