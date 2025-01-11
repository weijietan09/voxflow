"""文本归一化的单元测试。"""

import pytest

from voxflow.text.normalize import (
    collapse_whitespace,
    fullwidth_to_halfwidth,
    normalize,
    number_to_chinese,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "零"),
        (5, "五"),
        (10, "十"),
        (11, "十一"),
        (20, "二十"),
        (105, "一百零五"),
        (110, "一百一十"),
        (1000, "一千"),
        (10001, "一万零一"),
        (100000, "十万"),
        (100000000, "一亿"),
        (-3, "负三"),
    ],
)
def test_number_to_chinese(value, expected):
    assert number_to_chinese(value) == expected


def test_normalize_reads_numbers_in_chinese():
    assert normalize("我有3个苹果", language="zh") == "我有三个苹果"


def test_normalize_decimal():
    assert normalize("圆周率约3.14", language="zh") == "圆周率约三点一四"


def test_fullwidth_ascii_to_halfwidth():
    assert fullwidth_to_halfwidth("ＡＢＣ１２３") == "ABC123"


def test_fullwidth_space():
    assert fullwidth_to_halfwidth("你好　世界") == "你好 世界"


def test_collapse_whitespace():
    assert collapse_whitespace("  a\t\n b  ") == "a b"


def test_normalize_keeps_chinese():
    assert normalize("你好，世界！") == "你好，世界！"


def test_normalize_strips_zero_width():
    dirty = "\u4f60\u200b\u597d\ufeff"  # 你<ZWSP>好<BOM>
    assert normalize(dirty) == "\u4f60\u597d"
