"""轻量英文 G2P：小词典 + 逐字母回退，输出 ARPAbet 音素。

这不是高精度发音词典，而是一个自带、无需联网、无额外依赖的近似实现，
目的是打通中英混读链路。需要更高质量发音时可接入 ``g2p_en`` / CMUdict，
在 :class:`voxflow.text.frontend.TextFrontend` 里替换即可。
"""

from __future__ import annotations

import re

from voxflow.text.symbols import SIL

# 数字读法（英文单词）
_DIGIT_NAMES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

# 常见词的人工发音（ARPAbet，去掉重音数字）
_LEXICON: dict[str, list[str]] = {
    "a": ["AH"],
    "the": ["DH", "AH"],
    "hello": ["HH", "AH", "L", "OW"],
    "world": ["W", "ER", "L", "D"],
    "voice": ["V", "OY", "S"],
    "flow": ["F", "L", "OW"],
    "voxflow": ["V", "AA", "K", "S", "F", "L", "OW"],
    "zero": ["Z", "IH", "R", "OW"],
    "one": ["W", "AH", "N"],
    "two": ["T", "UW"],
    "three": ["TH", "R", "IY"],
    "four": ["F", "AO", "R"],
    "five": ["F", "AY", "V"],
    "six": ["S", "IH", "K", "S"],
    "seven": ["S", "EH", "V", "AH", "N"],
    "eight": ["EY", "T"],
    "nine": ["N", "AY", "N"],
}

# 二合字母（digraph）优先匹配
_DIGRAPHS: dict[str, list[str]] = {
    "ch": ["CH"],
    "sh": ["SH"],
    "th": ["TH"],
    "ph": ["F"],
    "wh": ["W"],
    "ck": ["K"],
    "ng": ["NG"],
    "qu": ["K", "W"],
    "oo": ["UW"],
    "ee": ["IY"],
    "ea": ["IY"],
    "ou": ["AW"],
    "ai": ["EY"],
    "ay": ["EY"],
    "oy": ["OY"],
    "ow": ["OW"],
}

# 单字母回退（粗糙）
_LETTER2PHONE: dict[str, list[str]] = {
    "a": ["AH"],
    "b": ["B"],
    "c": ["K"],
    "d": ["D"],
    "e": ["EH"],
    "f": ["F"],
    "g": ["G"],
    "h": ["HH"],
    "i": ["IH"],
    "j": ["JH"],
    "k": ["K"],
    "l": ["L"],
    "m": ["M"],
    "n": ["N"],
    "o": ["OW"],
    "p": ["P"],
    "q": ["K"],
    "r": ["R"],
    "s": ["S"],
    "t": ["T"],
    "u": ["AH"],
    "v": ["V"],
    "w": ["W"],
    "x": ["K", "S"],
    "y": ["Y"],
    "z": ["Z"],
}

_TOKEN_RE = re.compile(r"[a-z]+|[0-9]|[^\sa-z0-9]")


def word_to_phonemes(word: str) -> list[str]:
    """把一个英文单词转成 ARPAbet 音素列表。"""
    w = word.lower()
    if w in _LEXICON:
        return list(_LEXICON[w])
    phones: list[str] = []
    i = 0
    while i < len(w):
        digraph = w[i : i + 2]
        if digraph in _DIGRAPHS:
            phones.extend(_DIGRAPHS[digraph])
            i += 2
            continue
        phones.extend(_LETTER2PHONE.get(w[i], []))
        i += 1
    return phones


def english_to_tokens(text: str) -> list[str]:
    """把一段英文文本转成音素 token 序列，标点变成停顿。"""
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text.lower()):
        tok = match.group()
        if tok.isalpha():
            tokens.extend(word_to_phonemes(tok))
        elif tok.isdigit():
            tokens.extend(_LEXICON[_DIGIT_NAMES[int(tok)]])
        else:
            tokens.append(SIL)
    return tokens
