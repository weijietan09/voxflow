"""文本归一化：全角转半角、空白整理、标点规整。

数字读法在单独的一步里处理（见 :func:`normalize_numbers`）。
"""

from __future__ import annotations

import re

# 全角字符（含空格）到半角的映射
_FULLWIDTH_OFFSET = 0xFEE0
_FULLWIDTH_SPACE = 0x3000

# 归一化后统一保留的标点，其余标点会被替换成句号
_KEEP_PUNCT = "，。！？、；：…—,.!?;:"

_WHITESPACE_RE = re.compile(r"\s+")


def fullwidth_to_halfwidth(text: str) -> str:
    """把全角 ASCII 字符和全角空格转成半角。"""
    out = []
    for ch in text:
        code = ord(ch)
        if code == _FULLWIDTH_SPACE:
            out.append(" ")
        elif 0xFF01 <= code <= 0xFF5E:
            out.append(chr(code - _FULLWIDTH_OFFSET))
        else:
            out.append(ch)
    return "".join(out)


def collapse_whitespace(text: str) -> str:
    """把连续空白压成单个空格，并去掉首尾空白。"""
    return _WHITESPACE_RE.sub(" ", text).strip()


def normalize(text: str, language: str = "zh") -> str:
    """对输入文本做基础归一化。

    参数 ``language`` 目前只用于挑选后续步骤（数字读法等），
    这一层本身与语言无关。
    """
    text = fullwidth_to_halfwidth(text)
    text = collapse_whitespace(text)
    return text
