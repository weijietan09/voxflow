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
_DIGIT_RUN_RE = re.compile(r"\d+(?:\.\d+)?")
# 零宽字符：零宽空格 / 连接符 / BOM，粘贴文本里常见，需先清掉
_ZERO_WIDTH_RE = re.compile("[\u200b\u200c\u200d\ufeff]")

_CN_DIGITS = "零一二三四五六七八九"
_CN_SMALL_UNITS = ["", "十", "百", "千"]
_CN_BIG_UNITS = ["", "万", "亿", "兆"]


def _section_to_chinese(sec: int) -> str:
    """把 0-9999 的整数读成中文（不含大单位）。"""
    out = ""
    pos = 0
    while sec > 0:
        digit = sec % 10
        if digit != 0:
            out = _CN_DIGITS[digit] + _CN_SMALL_UNITS[pos] + out
        elif out and not out.startswith("零"):
            out = "零" + out
        sec //= 10
        pos += 1
    return out


def number_to_chinese(n: int) -> str:
    """把整数读成中文，例如 105 -> 一百零五、10001 -> 一万零一。"""
    if n == 0:
        return "零"
    negative = n < 0
    n = abs(n)

    sections: list[int] = []
    while n > 0:
        sections.append(n % 10000)
        n //= 10000

    result = ""
    for i in range(len(sections) - 1, -1, -1):
        sec = sections[i]
        if sec == 0:
            if result and not result.endswith("零"):
                result += "零"
            continue
        if result and sec < 1000 and not result.endswith("零"):
            result += "零"
        result += _section_to_chinese(sec) + (_CN_BIG_UNITS[i] if i > 0 else "")

    result = re.sub("零+", "零", result).rstrip("零")
    if result.startswith("一十"):
        result = result[1:]
    return ("负" if negative else "") + result


def _read_number_token(token: str) -> str:
    """读一个数字串，支持一位小数点（3.14 -> 三点一四）。"""
    if "." in token:
        int_part, frac_part = token.split(".", 1)
        head = number_to_chinese(int(int_part)) if int_part else "零"
        tail = "".join(_CN_DIGITS[int(d)] for d in frac_part)
        return f"{head}点{tail}"
    return number_to_chinese(int(token))


def normalize_numbers(text: str) -> str:
    """把文本里的阿拉伯数字替换成中文读法。"""
    return _DIGIT_RUN_RE.sub(lambda m: _read_number_token(m.group()), text)


def normalize(text: str, language: str = "zh") -> str:
    """对输入文本做基础归一化。

    对中文（``language`` 以 ``zh`` 开头）会额外把阿拉伯数字转成中文读法。
    """
    text = _ZERO_WIDTH_RE.sub("", text)
    text = fullwidth_to_halfwidth(text)
    text = collapse_whitespace(text)
    if language.lower().startswith("zh"):
        text = normalize_numbers(text)
    return text


def fullwidth_to_halfwidth(text: str) -> str:
    """把全角字母 / 数字 / 空格转成半角，保留全角标点。

    中文标点（，。！？ 等）承担停顿与语气信息，交给后续步骤处理，
    这里不把它们折叠成 ASCII 标点。
    """
    out = []
    for ch in text:
        code = ord(ch)
        if code == _FULLWIDTH_SPACE:
            out.append(" ")
        elif 0xFF10 <= code <= 0xFF19 or 0xFF21 <= code <= 0xFF3A or 0xFF41 <= code <= 0xFF5A:
            out.append(chr(code - _FULLWIDTH_OFFSET))
        else:
            out.append(ch)
    return "".join(out)


def collapse_whitespace(text: str) -> str:
    """把连续空白压成单个空格，并去掉首尾空白。"""
    return _WHITESPACE_RE.sub(" ", text).strip()
