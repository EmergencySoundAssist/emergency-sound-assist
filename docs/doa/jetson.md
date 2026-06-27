# Jetson 배포 런북 (DoA)

대상: **Jetson Orin Nano (JetPack / aarch64)** + ReSpeaker USB 4-Mic Array.

세 단계로 위험을 줄인다 — **하드웨어 없이 먼저(데모) → 실물 오디오(udev 불필요) → (옵션) 경량 XVF(udev 필요).**

| 경로 | 명령 | 하드웨어 | udev | 추가 의존성 |
|------|------|----------|------|------------|
| **데모(검증)** | `doa-multilive --demo` | ❌ 불필요 | ❌ | pyroomacoustics |
| **오디오 SRP/MUSIC (주)** | `doa-multilive --no-led` | ReSpeaker | ❌ | pyroomacoustics + libportaudio2 |
| **경량 XVF (옵션)** | `doa-live` | ReSpeaker | ✅ **필요** | 가벼움(pyusb) |

> ⚠️ **conda 금지.** JetPack/aarch64 에선 Anaconda 가 맞지 않는다. 반드시 `python3 -m venv` 사용.
> 방향만 필요하고 LED를 안 쓰면 **오디오 경로(udev 불필요)** 로 충분하다 — 다른 분류/접근 브랜치가
> Jetson에서 잘 돌던 것과 같은 접근이다. udev가 필요한 건 XVF 레지스터·LED 같은 USB 제어뿐.

---

## 0. 코드 가져오기 (detached HEAD 주의)
```bash
cd ~/emergency-sound-assist
git fetch origin && git switch doa-jamin   # origin/doa-jamin 직접 체크아웃은 detached HEAD → pull 안 됨
git rev-parse --short HEAD                  # 기대 커밋인지 확인 (옛 코드 실행 사고 방지)
```

## 1. 가상환경 + 설치
```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -U pip
sudo apt install -y libportaudio2          # sounddevice(오디오) 런타임
# pyroomacoustics 가 소스 빌드로 떨어질 수 있어 빌드 도구 미리:
sudo apt install -y build-essential python3-dev
pip install -e ".[multisource]"            # numpy, scipy, sounddevice, pyroomacoustics
```
> pyroomacoustics aarch64 휠이 없어 빌드가 오래/실패하면, scipy 가 먼저 import 되는지 확인하고
> piwheels/JetPack 제공 휠을 우선 사용한다.

## 2. 하드웨어 없이 먼저 검증 (장치·udev 불필요)
```bash
python -m doa.multi_live --demo            # 또는 doa-multilive --demo
```
합성 음원을 한 바퀴 돌리며 입력각 vs 추정각을 출력하고, 끝에 `N/48 프레임 오차 ≤10°` 를 보여준다.
**이게 통과하면 "설치(pyroomacoustics aarch64 포함)·SRP 연산이 이 Jetson에서 동작"** 이 확인된 것 —
남은 변수는 실물 마이크 연결뿐이다.

## 3. 실물 오디오 경로 — 주 경로 (udev 불필요)
```bash
python -c "import sounddevice as sd; print(sd.query_devices())"   # 6채널 'ReSpeaker' 인덱스 확인
python -m doa.multi_live --no-led                                 # 자동 탐지
python -m doa.multi_live --no-led --device 2                      # 이름 매칭 실패 시 인덱스 지정
```
오디오 캡처는 ALSA(snd-usb-audio)로 처리돼 **별도 USB 권한이 필요 없다.** 방향이 안 뜨고
`ReSpeaker(6채널)를 못 찾음` 이면 `query_devices()` 의 6채널 장치 인덱스를 `--device` 로 직접 지정.

## 4. (옵션) 경량 XVF 경로 — udev 필요
LED나 ReSpeaker 자체 DoA(`doa.live`, pyroomacoustics 불필요)를 쓰려면 **USB 제어 권한**이 필요하다.

```bash
sudo apt install -y libusb-1.0-0
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="2886", ATTRS{idProduct}=="0018", MODE="0666"' \
  | sudo tee /etc/udev/rules.d/99-respeaker.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
# ReSpeaker 재플러그 후:
lsusb | grep 2886        # 2886:0018 보이면 인식 OK
python -m doa.live       # 또는 doa-live
```

### 트러블슈팅 — `장치 없음` / `angle=None`
1. **권한 vs stale 코드 구분**: `sudo $(which python) -m doa.live` 로 떠보기.
   - sudo 면 되고 일반은 안 되면 → **udev 권한 문제** (위 규칙·재플러그 재확인).
   - sudo 로도 안 되면 → 연결/`lsusb`/`libusb` 문제.
2. estimator 가 `[doa] ReSpeaker 방향 읽기 실패: ...` 를 stderr 로 찍으면 그 메시지가 원인.
3. **그냥 udev 없이 쓰려면 3번(오디오 경로)** 으로. 방향 데이터는 거기서도 나온다.

## 5. (참고) systemd 상시 구동
```ini
[Service]
WorkingDirectory=/home/<user>/emergency-sound-assist
ExecStart=/home/<user>/emergency-sound-assist/.venv/bin/doa-multilive --no-led
Restart=on-failure
```
`pip install -e .` 로 설치돼 있으면 `doa-multilive`(console script)가 CWD 와 무관하게 동작하므로
`python -m doa.x` 의 '레포 루트에서만' 제약을 피할 수 있다.
