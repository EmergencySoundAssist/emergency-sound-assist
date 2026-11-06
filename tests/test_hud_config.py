"""HudConfig 기본값 — pygame 불필요(순수)."""

from hud import HudConfig


def test_hudconfig_defaults():
    c = HudConfig()
    assert c.fullscreen is True
    assert c.reflect is False
    assert c.fps == 30
    assert c.width == 1280 and c.height == 360   # 윈드실드용 얇은 띠
    assert c.font_path is None
