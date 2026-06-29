# DoA 실행 명령어 모음

방향추정(DoA) 코드를 돌릴 때 쓰는 명령어 정리. **항상 `python` 사용**(이 프로젝트 conda 환경 기준 — `python3`는 시스템 파이썬을 가리킬 수 있음).

> **설정은 [`doa/config.py`](../../doa/config.py) 한 곳에서 관리.** 매번 플래그를 치기 싫으면
> config.py 값(hop/hold/algo/num/led + 스무딩 `SMOOTH`/`CONF_MIN` + 보정값)을 바꾸면 된다.
> 그래도 아래 CLI 플래그를 주면 그 실행에 한해 덮어쓴다. (예: config 에서 `LED=True` 면 `--led` 없이도 켜짐)

> **Jetson 배포는 별도 런북** → [jetson.md](./jetson.md) (venv·udev·경량/무거운 경로).

---

## 0. 환경 (최초 1회 + 작업 시작 시)
```bash
# 최초 1회 (Mac/일반)
python3 -m venv .venv && source .venv/bin/activate && pip install -U pip
pip install -r requirements.txt       # 의존성 설치
#   ※ Jetson(aarch64)은 conda 불가 → 위 venv 사용. 상세: jetson.md

# 작업 시작할 때마다
source .venv/bin/activate
```
실행은 **레포 루트에서 `python -m doa.<모듈>`** (모듈 실행 — `python doa/live.py` 는 import 깨짐).

## 1. 테스트 (하드웨어 불필요)
```bash
python -m pytest -q                           # 전체 (56개)
python -m pytest tests/test_multi_source.py   # 다중 peak + 신뢰도(spectrum_confidence)
python -m pytest tests/test_tracking.py       # 시간 다수결(circular_median, DirectionTracker)
python -m pytest tests/test_led_ring.py       # LED 매핑/payload
```

## 2. 단일 방향 (1단계 — ReSpeaker 자체 DoA, 경량)
```bash
python -m doa.estimator        # 각도→4방향 매핑 데모 (장치 있으면 실측 1회)
python -m doa.live             # 자체 DoA 실시간 폴링 (numpy+pyusb 만, Jetson 경량 경로)
```

## 3. 다중 방향 — 1회 캡처
```bash
python -m doa.multi_source     # 1초 캡처 → 동시 여러 방향 출력 (pyroomacoustics 필요)
```

## 4. 다중 방향 — 실시간 (주력)
```bash
python -m doa.multi_live                              # 기본 (단일 음원 + 스무딩)
python -m doa.multi_live --led                        # + LED 링 점등
python -m doa.multi_live --no-smooth                  # 스무딩 끄고 옛 즉시반응과 비교
python -m doa.multi_live --num 2 --algo music         # 동시 2개 (MUSIC) — 유령 자동 제거
python -m doa.multi_live --device 2                   # 입력 장치 인덱스 직접 지정
```
> **다중 감지**: 기본은 단일(`NUM_SRC=1`, 신뢰성·Jetson 가벼움). `--num 2 --algo music` 으로 켜면
> primary 기준 ~180° 가짜 2번째(전후 유령)는 자동 제거(`PHANTOM_TOL_DEG`)된다. 항상 다중으로
> 쓰려면 `config.py` 의 `NUM_SRC=2`, `ALGO="MUSIC"` 두 줄만 바꾸면 됨. 단 4-mic라 동시 분리는
> best-effort(같은 대역 사이렌끼리는 못 가름).

### 하드웨어 없이 검증 — `--demo`
ReSpeaker 없이 **합성 음원**으로 전체 파이프라인(SRP + 스무딩 + 매핑)을 돌린다. 새 기기(특히
**Jetson aarch64**)에서 "설치(pyroomacoustics)·연산이 도는가"를 장치/udev 없이 즉시 확인:
```bash
python -m doa.multi_live --demo            # 음원을 한 바퀴 돌리며 입력각 vs 추정각 출력
python -m doa.multi_live --demo --algo music
```
끝에 `N/48 프레임 오차 ≤10°` 가 뜨면 정상. (절대 각도 보정은 실물 `doa.diag` 로.)
- **방향이 확정되면** 한 줄 실시간 갱신, 동시 2개 이상이면 `#1 ★다중(2) ...` 영구 로그.
- 신뢰도가 낮으면(반사/잡음) **`방향 불확실 (감지됨 conf=..)`** — 틀린 화살표 대신 안전쪽.

### 견고화 — 시간 다수결 + 신뢰도 게이팅 (기본 ON)
반사·터널에서 방향이 ±180° 튀는 것을 억제한다. 자세히는 [`doa/tracking.py`](../../doa/tracking.py).
- **검출(소리 왔다)은 즉시** — 화살표만 안정화하므로 검출 지연은 0.
- **`--conf-min`**: 주엽 우월도(주엽÷반대편 반원 최대) 하한. 미만이면 '방향 불확실'.
  - 1≈반대편과 맞먹음(튐), 클수록 한 방향 뚜렷. 기본 2.0. `--conf-min 0` 으로 conf 분포 관찰.
- **`--no-smooth`**: 게이팅·다수결 끄고 매 프레임 즉시 표시(비교용).

### 옵션
| 옵션 | 기본 | 의미 |
|---|---|---|
| `--led` / `--no-led` | off | 감지 방향을 LED 링에 점등 |
| `--smooth` / `--no-smooth` | on | 시간 다수결+신뢰도 게이팅(±180° 튐 억제) |
| `--conf-min` | 2.0 | 이 신뢰도(주엽 우월도) 미만이면 '방향 불확실' |
| `--algo` | SRP | `SRP`(가벼움) / `MUSIC`(다중 분리 유리, 무거움) |
| `--num` | 1 | 최대 동시 음원 수. 1=MVP(유령 2번째 억제), 동시 다중은 2~3(4-mic 최대 3) |
| `--window` | 0.4 | 분석 창(초) — 길수록 방향 해상도↑ |
| `--hop` | 0.1 | 갱신 간격(초) — 짧을수록 빠름(≈10Hz). `⚠느림` 뜨면 늘릴 것 |
| `--hold` | 1.0 | 감지 후 LED 유지 시간(초) — 소리 끊겨도 이만큼 점등 유지 |
| `--height` | 0.5 | 2번째 peak 임계(최대 대비). 낮추면 약한 음원도 잡음 |
| `--min-sep` | 30 | 두 음원 최소 각도 간격(도). 낮추면 가까운 음원 분리 |
| `--threshold` | 0.01 | 이 RMS 미만이면 '조용함' |

### 자주 쓰는 조합
```bash
python -m doa.multi_live --led --hop 0.05 --window 0.3       # 더 빠른 반응(≈20Hz)
python -m doa.multi_live --num 2 --algo music --height 0.2   # 2개 동시 감지 도전
python -m doa.multi_live --led --conf-min 0                  # 게이팅 끄고 conf 값 관찰(튜닝)
```

## 5. 방향 진단/보정 (단일 음원)
```bash
python -m doa.diag --truth front --led    # 정면에 소리 → raw/차량각/방향/conf + 원형통계
python -m doa.diag --truth rear           # 후방 등 4방향 각각 찍어 보정 확정
```
종료(Ctrl+C) 시 원형통계로 **(A) 구경 한계(통계적 튐)** vs **(B) 규약·보정 오차(일정 틀어짐)**를 자동 판정.
`--truth` 와 60°↑ 차이면 (B) — `config` 의 `REAR_RAW_DEG`/`MIRROR` 로 상수 보정.

## 6. LED 단독 점검 (오디오 무관)
```bash
python -m doa.led_ring         # LED 한 칸씩 회전 — 링 동작 확인
```

---

## 트러블슈팅
| 증상 | 해결 |
|---|---|
| `No module named ...` | venv 활성화(`source .venv/bin/activate`) + `pip install -r requirements.txt` 했는지. **레포 루트에서 `python -m doa.x`** 로 실행했는지 |
| `Invalid number of channels` | 기본 입력이 ReSpeaker가 아님 → 자동 탐지하지만 `sd.query_devices()`로 6채널 장치 확인 |
| 계속 `방향 불확실` | `--conf-min 0` 으로 conf 분포 보고 임계 조정. 단일 음원이면 `--num 1` (유령 2번째 제거) |
| 방향이 ±180° 반대로 튐 | 스무딩이 억제(기본 ON). 그래도 심하면 `--conf-min` 올리고, 단일 음원은 `--num 1` |
| LED `Access denied` | `sudo python -m doa.led_ring` 또는 udev 규칙(리눅스/Jetson → [jetson.md](./jetson.md)) |
| `⚠느림` 표시 | `--window` 줄이기 / `--algo srp` / `--hop` 늘리기 |
| 방향은 맞는데 LED 위치가 회전됨 | `config.LED_OFFSET` 조정(한 칸=30°) |
| 좌/우가 반대 | `config.MIRROR` 토글 ([direction-mapping.md](./direction-mapping.md)) |
| `장치 없음`/`angle=None` (Jetson) | USB 권한(udev)/libusb → [jetson.md](./jetson.md) 트러블슈팅 |
