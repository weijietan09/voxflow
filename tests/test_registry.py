"""针对组件注册表的单元测试。"""

import pytest

from voxflow.exceptions import ConfigError
from voxflow.registry import Registry


def test_register_and_build():
    reg: Registry[str] = Registry("demo")

    @reg.register()
    class Widget:
        def __init__(self, size: int) -> None:
            self.size = size

    assert "Widget" in reg
    obj = reg.build("Widget", 3)
    assert obj.size == 3
    assert reg.keys() == ["Widget"]


def test_register_with_explicit_key():
    reg: Registry[int] = Registry("nums")
    reg.register("answer")(lambda: 42)
    assert reg.build("answer") == 42


def test_duplicate_registration_raises():
    reg: Registry[int] = Registry("nums")
    reg.register("a")(lambda: 1)
    with pytest.raises(ConfigError):
        reg.register("a")(lambda: 2)


def test_missing_key_raises():
    reg: Registry[int] = Registry("nums")
    with pytest.raises(ConfigError):
        reg.get("ghost")
