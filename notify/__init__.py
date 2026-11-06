"""전송(BLE) 패키지. `notify.BleSender` 로 폰에 (AlertEvent, info) 경보 전송."""
from .ble_sender import BleSender

__all__ = ["BleSender"]
