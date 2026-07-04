"""极简的组件注册表。

用字符串名字注册类或构造函数，方便直接从 YAML 配置按名字实例化
模型组件，而不必在代码里堆一长串 ``if / elif``。
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Generic, TypeVar

from voxflow.exceptions import ConfigError

T = TypeVar("T")


class Registry(Generic[T]):
    """名字到可调用对象的映射，可当装饰器用。"""

    def __init__(self, name: str) -> None:
        self._name = name
        self._store: dict[str, Callable[..., T]] = {}

    def register(
        self, key: str | None = None
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """把一个类 / 函数登记到注册表；不传 ``key`` 时使用其 ``__name__``。"""

        def decorator(obj: Callable[..., T]) -> Callable[..., T]:
            name = key or getattr(obj, "__name__", None)
            if not name:
                raise ConfigError("注册对象缺少名字，且未显式提供 key")
            if name in self._store:
                raise ConfigError(f"{self._name} 中已存在同名组件: {name!r}")
            self._store[name] = obj
            return obj

        return decorator

    def get(self, key: str) -> Callable[..., T]:
        if key not in self._store:
            raise ConfigError(
                f"{self._name} 中找不到 {key!r}，已注册: {sorted(self._store)}"
            )
        return self._store[key]

    def build(self, key: str, *args: object, **kwargs: object) -> T:
        """按名字取出并实例化。"""
        return self.get(key)(*args, **kwargs)

    def keys(self) -> list[str]:
        return sorted(self._store)

    def __contains__(self, key: object) -> bool:
        return key in self._store

    def __iter__(self) -> Iterator[str]:
        return iter(self._store)

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        return f"Registry({self._name!r}, {len(self)} 项)"
