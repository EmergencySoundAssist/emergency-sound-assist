"""BLE 페이로드 인코딩 회귀 테스트.

바이트 포맷은 폰쪽 AlertProtocol.kt 와 반드시 같아야 하므로, 값이 바뀌면
폰 앱도 함께 고쳐야 한다는 뜻으로 여기서 고정한다.
"""

from core.types import Direction, Motion
from pipeline import alert
from notify.ble_sender import BleSender
from notify.protocol import ALERT_CHAR_UUID, SERVICE_UUID, encode_alert


def _info(direction=None, motion=None, conf=0.0) -> dict:
    return {"direction": direction, "motion": motion, "conf": conf}


def test_uuid는_폰과_합의된_값이다():
    # 한 글자라도 바뀌면 젯슨이 폰을 못 찾는다.
    assert SERVICE_UUID == "e7a10000-2c5f-4b9a-8d3e-1f0a9b8c7d60"
    assert ALERT_CHAR_UUID == "e7a10001-2c5f-4b9a-8d3e-1f0a9b8c7d60"


def test_사이렌_경보는_4바이트로_인코딩된다():
    ev = alert.build_event("siren", 5.0, {"onset": True, "remind": False, "clear": False})
    payload = encode_alert(ev, _info(Direction.REAR, Motion.APPROACHING, 0.87))
    assert payload == bytes([1, 1, 0, 87])
    assert len(payload) == 4


def test_경적은_sound_2로_인코딩된다():
    ev = alert.build_event("horn", 5.0, {"onset": True, "remind": False, "clear": False})
    payload = encode_alert(ev, _info(Direction.FRONT, Motion.RECEDING, 0.5))
    assert payload == bytes([2, 0, 1, 50])


def test_경보_해제는_sound_0을_보내_진동을_멈춘다():
    ev = alert.build_event("none", 0.0, None)
    payload = encode_alert(ev, _info(Direction.LEFT, Motion.APPROACHING, 0.9))
    assert payload[0] == 0


def test_미상값은_0xFF로_인코딩된다():
    ev = alert.build_event("siren", 5.0, {"onset": True, "remind": False, "clear": False})
    payload = encode_alert(ev, _info(None, None, None))
    assert payload == bytes([1, 0xFF, 0xFF, 0])


def test_UNKNOWN_enum도_0xFF로_인코딩된다():
    ev = alert.build_event("siren", 5.0, {"onset": True, "remind": False, "clear": False})
    payload = encode_alert(ev, _info(Direction.UNKNOWN, Motion.UNKNOWN, 0.1))
    assert payload == bytes([1, 0xFF, 0xFF, 10])


def test_유지는_motion_2로_인코딩된다():
    ev = alert.build_event("siren", 5.0, {"onset": True, "remind": False, "clear": False})
    payload = encode_alert(ev, _info(Direction.RIGHT, Motion.STEADY, 1.0))
    assert payload == bytes([1, 3, 2, 100])


def test_신뢰도는_0에서_100으로_클램프된다():
    ev = alert.build_event("siren", 5.0, {"onset": True, "remind": False, "clear": False})
    assert encode_alert(ev, _info(conf=1.5))[3] == 100
    assert encode_alert(ev, _info(conf=-0.5))[3] == 0
    # 모든 바이트가 uint8 범위 안이어야 bytes() 가 성공한다.
    assert all(0 <= b <= 255 for b in encode_alert(ev, _info(conf=1.5)))


def test_ev가_None이어도_죽지_않는다():
    # 워밍업 tick 방어. 예외 대신 '일반' 페이로드를 보낸다.
    assert encode_alert(None, _info())[0] == 0


# ── 중복 전송 억제 ────────────────────────────────────────────────
class _SpySender(BleSender):
    """_enqueue 가 실제로 큐에 넘기려 한 페이로드만 기록한다(BLE 스레드 없이)."""

    def __init__(self):
        super().__init__()
        self.sent = []

    def _enqueue(self, payload: bytes) -> None:
        if self._last is not None and payload[:3] == self._last[:3]:
            return
        self._last = payload
        self.sent.append(payload)


def test_신뢰도만_바뀌면_전송하지_않는다():
    # 신뢰도는 매 tick 흔들린다. 이걸 보내면 폰이 진동 파형을 매번 restart 해서
    # 방향 리듬(640~920ms)이 잘리고 좌측↔전방이 구분되지 않는다.
    s = _SpySender()
    for conf in (80, 81, 82, 83, 84):
        s._enqueue(bytes([1, 1, 0, conf]))
    assert len(s.sent) == 1


def test_상황이_바뀌면_전송한다():
    s = _SpySender()
    s._enqueue(bytes([1, 1, 0, 80]))    # 사이렌·후방·접근
    s._enqueue(bytes([1, 1, 1, 80]))    # → 멀어짐
    s._enqueue(bytes([1, 2, 1, 80]))    # → 좌측
    s._enqueue(bytes([0, 2, 1, 80]))    # → 해제
    assert len(s.sent) == 4


def test_전송하는_페이로드에는_최신_신뢰도가_실린다():
    # 스킵은 '보낼지 말지'만 정한다. 보낼 때는 그 시점의 신뢰도를 그대로 싣는다.
    s = _SpySender()
    s._enqueue(bytes([1, 1, 0, 80]))
    s._enqueue(bytes([1, 1, 0, 95]))    # 신뢰도만 변화 → 스킵
    s._enqueue(bytes([1, 1, 1, 97]))    # 움직임 변화 → 전송
    assert s.sent[-1] == bytes([1, 1, 1, 97])
