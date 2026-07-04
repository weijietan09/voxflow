"""汉语拼音 G2P：把汉字转成 声母 / 韵母 / 声调 token 序列。

依赖 ``pypinyin`` 拿到带声调的拼音，再切成与符号表一致的 token：
声母、韵母各占一个 token，声调（1-4，轻声记 5）单独占一个 token。
"""

from __future__ import annotations

import re

from pypinyin import Style, lazy_pinyin

from voxflow.text.symbols import SIL

# 两字母声母需要优先匹配，避免把 "zh" 误切成 "z"
_TWO_CHAR_INITIALS = ("zh", "ch", "sh")
_SINGLE_INITIALS = set("bpmfdtnlgkhjqxrzcsyw")
_SYLLABLE_RE = re.compile(r"^[a-z]+[1-5]?$")


def _normalize_syllable(syllable: str) -> str:
    """把 ü 的各种写法统一成 v，方便和符号表对齐。"""
    return syllable.replace("ü", "v").replace("u:", "v")


def split_initial_final(base: str) -> tuple[str, str]:
    """把不带声调的音节切成 (声母, 韵母)；零声母时声母为空串。"""
    if base[:2] in _TWO_CHAR_INITIALS:
        return base[:2], base[2:]
    if base[0] in _SINGLE_INITIALS:
        return base[0], base[1:]
    return "", base


def chinese_to_tokens(text: str) -> list[str]:
    """把一段（以汉字为主的）文本转成 token 序列。

    无法识别的字符（标点等）会退化成一个停顿 token :data:`SIL`。
    """
    tokens: list[str] = []
    syllables = lazy_pinyin(
        text,
        style=Style.TONE3,
        neutral_tone_with_five=True,
        errors=lambda chars: list(chars),
    )
    for raw in syllables:
        syl = _normalize_syllable(raw)
        if not _SYLLABLE_RE.match(syl):
            tokens.append(SIL)
            continue
        if syl[-1].isdigit():
            tone, base = syl[-1], syl[:-1]
        else:
            tone, base = "5", syl
        if not base:
            tokens.append(SIL)
            continue
        initial, final = split_initial_final(base)
        if initial:
            tokens.append(initial)
        if final:
            tokens.append(final)
        tokens.append(tone)
    return tokens
