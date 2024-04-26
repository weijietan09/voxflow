"""文本归一化的单元测试。"""

from voxflow.text.normalize import (
    collapse_whitespace,
    fullwidth_to_halfwidth,
    normalize,
)


def test_fullwidth_ascii_to_halfwidth():
    assert fullwidth_to_halfwidth("ＡＢＣ１２３") == "ABC123"


def test_fullwidth_space():
    assert fullwidth_to_halfwidth("你好　世界") == "你好 世界"


def test_collapse_whitespace():
    assert collapse_whitespace("  a\t\n b  ") == "a b"


def test_normalize_keeps_chinese():
    assert normalize("你好，世界！") == "你好，世界！"
