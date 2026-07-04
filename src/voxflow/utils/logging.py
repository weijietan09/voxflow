"""统一的 logger 获取入口。

第一次调用时给 ``voxflow`` 这个根 logger 挂一个 stderr handler，
之后所有子 logger 复用同一份格式，避免重复输出。
"""

from __future__ import annotations

import logging

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"
_configured = False


def get_logger(name: str = "voxflow", level: int = logging.INFO) -> logging.Logger:
    """返回一个带统一格式的 logger。"""
    global _configured
    root = logging.getLogger("voxflow")
    if not _configured:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
        root.addHandler(handler)
        root.setLevel(level)
        root.propagate = False
        _configured = True
    return logging.getLogger(name)
