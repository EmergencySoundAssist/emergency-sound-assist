# Jetson 배포 런북 (DoA)

대상: **Jetson Orin Nano (JetPack / aarch64)** + ReSpeaker USB 4-Mic Array.

핵심 원칙 — **경량 경로부터** 올린다. 방향만 필요하면 무거운 의존성(pyroomacoustics/sounddevice)
없이 동작한다. 다중 음원(SRP/MUSIC)은 검증 후 옵션으로 추가한다.

| 경로 | 모듈 | 추가 의존성 | 용도 |
|------|------|------------|------|
| **경량** | `doa.live` | numpy, pyusb (+libusb, udev) | XVF 자체 DoA 방향(전/후/좌/우) |
| **무거움(옵션)** | `doa.multi_live`, `doa.diag` | + scipy, sounddevice(+libportaudio2), pyroomacoustics | 다중 음원 SRP/MUSIC + LED |

> ⚠️ **conda 금지.** JetPack/aarch64 에선 Anaconda 가 맞지 않는다. 반드시 `python3 -m venv` 사용.

---

## 0. 코드 가져오기 (detached HEAD 주의)

```bash
cd ~/emergency-sound-assist
git fetch origin
git switch doa-jamin          # 로컬 브랜치 위로 (origin/doa-jamin 직접 체크아웃 = detached HEAD → pull 안 됨)
git rev-parse --short HEAD    # 기대한 커밋인지 확인 (옛 코드 실행 사고 방지)
```

## 1. 가상환경

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
```

## 2. 경량 경로 — 시스템 패키지 + 설치

```bash
sudo apt update
sudo apt install -y libusb-1.0-0        # pyusb 백엔드 (없으면 'No backend available')
pip install -e .                        # numpy + pyusb 만 (가벼움)
```

## 3. ReSpeaker USB 권한 (udev) — **angle=None 의 진짜 원인**

권한이 없으면 방향 읽기가 조용히 실패해 `angle=None` / `UNKNOWN` 으로 떨어진다.

```bash
sudo tee /etc/udev/rules.d/99-respeaker.rules >/dev/null <<'EOF'
SUBSYSTEM=="usb", ATTRS{idVendor}=="2886", ATTRS{idProduct}=="0018", MODE="0666"
EOF
sudo udevadm control --reload-rules && sudo udevadm trigger
# ReSpeaker 를 뽑았다 다시 꽂는다
lsusb | grep 2886                        # 2886:0018 보이면 인식 OK
```

## 4. 경량 실행

```bash
python -m doa.live      # 또는 설치됐으니: doa-live
```

각도가 4방향으로 흐르면 성공. `장치 없음...` 이 뜨면 아래 트러블슈팅.

### 트러블슈팅 — angle=None / "장치 없음"
1. **권한인지 stale 코드인지 가르기**: 한 번 `sudo $(which python) -m doa.live` 로 떠보기.
   - sudo 면 되고 일반은 안 되면 → **udev 권한 문제** (3번 재확인, 재플러그).
   - sudo 로도 안 되면 → 연결/`lsusb`/`libusb` 또는 코드 문제.
2. **stale 코드**: `git rev-parse --short HEAD` 가 기대 커밋인지, `python -m doa.live` 를 **레포 루트에서** 실행했는지(설치했으면 `doa-live` 는 CWD 무관).
3. estimator 가 "ReSpeaker 방향 읽기 실패: ..." 를 stderr 로 찍으면 그 메시지(USBError/NoBackend)가 원인.

## 5. (옵션) 무거운 경로 — 다중 음원 SRP/MUSIC

```bash
sudo apt install -y libportaudio2                 # sounddevice(PortAudio) 런타임
# pyroomacoustics 가 소스 빌드로 떨어질 수 있음 → 빌드 도구 미리:
sudo apt install -y build-essential python3-dev
pip install -e ".[multisource]"                   # scipy, sounddevice, pyroomacoustics
```

오디오 장치(6채널 ReSpeaker) 이름 확인 후 실행:

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"   # 6채널 'ReSpeaker' 인덱스 확인
python -m doa.multi_live --led                                    # 또는 doa-multilive --led
```

> pyroomacoustics 가 aarch64 휠이 없어 소스 빌드가 오래 걸리거나 실패하면: scipy 가 먼저 import 되는지
> 확인하고, 가능하면 piwheels/JetPack 제공 휠을 우선 사용한다.

## 6. (참고) systemd 로 상시 구동

```ini
[Service]
WorkingDirectory=/home/<user>/emergency-sound-assist
ExecStart=/home/<user>/emergency-sound-assist/.venv/bin/doa-live
Restart=on-failure
```

`pip install -e .` 로 설치돼 있으면 `doa-live`(console script)가 CWD 와 무관하게 동작하므로,
`python -m doa.live` 의 '레포 루트에서만' 제약을 피할 수 있다.
