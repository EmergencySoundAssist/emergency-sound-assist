# Jetson 방향 추정 배포

대상은 Jetson Orin Nano와 ReSpeaker USB 6채널 입력이다. 통합 파이프라인은 ch1~4 raw
마이크를 SRP-PHAT에 사용한다.

## 설치

```bash
sudo apt update
sudo apt install -y libportaudio2 libusb-1.0-0 build-essential python3-dev

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

`pyroomacoustics`가 aarch64에서 소스 빌드되면 설치 시간이 길 수 있다. 방향 오디오 경로는
`numpy`, `scipy`, `sounddevice`, `pyroomacoustics`가 필요하고, 자체 DoA와 LED는 `pyusb`와
`libusb`가 추가로 필요하다.

## 하드웨어 없는 확인

```bash
python -m doa.multi_live --demo
python -m doa.multi_live --demo --algo MUSIC
```

이 단계가 통과하면 오디오 장치와 무관하게 pyroomacoustics 연산과 방향 매핑이 동작한다.

## ReSpeaker 오디오 방향

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
python -m doa.multi_live --no-led
python -m doa.multi_live --no-led --device 2
```

이 경로는 ALSA 오디오로 ch1~4를 읽으므로 일반적으로 별도 USB 제어 권한이 필요 없다.

## 자체 DoA와 LED 권한

`doa.live`와 `--led`는 USB 제어를 사용한다. 권한 오류가 있으면 ReSpeaker의 실제 vendor/product
ID를 `lsusb`로 확인한 뒤 udev 규칙을 만든다. 장치 ID를 확인하지 않고 문서의 예시 값을 그대로
복사하지 않는다.

```bash
lsusb
sudo udevadm control --reload-rules
sudo udevadm trigger
```

규칙 적용 후 ReSpeaker를 다시 연결하고 확인한다.

```bash
python -m doa.live
python -m doa.multi_live --led
```

## 통합 실행과 보정

```bash
python main.py --mic --channels 6 --device 2
python -m doa.diag --truth front --device 2
python -m doa.diag --truth rear --device 2
python -m doa.diag --truth left --device 2
python -m doa.diag --truth right --device 2
```

차량 장착 후 `doa/config.py`의 `REAR_RAW_DEG`, `MIRROR`, `MIC_RADIUS_M`, `CONF_MIN`,
`LED_OFFSET`을 실측으로 확정한다.

## 문제 해결

| 증상 | 확인할 것 |
|---|---|
| ReSpeaker를 못 찾음 | `sd.query_devices()`에서 6채널 입력 번호 확인 후 `--device N` |
| `Invalid number of channels` | 다른 마이크를 선택했거나 ReSpeaker 입력 채널 설정 오류 |
| 계속 방향 불확실 | `--conf-min 0`으로 분포 확인 후 실제 환경에서 임계 조정 |
| 좌우가 반대 | `doa/config.py`의 `MIRROR` |
| 일정 각도로 회전됨 | `REAR_RAW_DEG`, 장착 방향, LED면 `LED_OFFSET` |
| LED/자체 DoA Access denied | libusb와 udev 권한 확인 |

세부 실행 옵션은 [DoA 실행 문서](running.md)를 참고한다.
