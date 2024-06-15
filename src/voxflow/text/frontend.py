"""统一的中英双语文本前端。

处理流程：

1. 归一化（全角转半角、数字读法……）；
2. 按语言把文本切成中文 / 英文片段；
3. 各自做 G2P（拼音 / ARPAbet）；
4. 拼接 token 并映射成符号表 id。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from voxflow.text.english import english_to_tokens
from voxflow.text.normalize import normalize
from voxflow.text.pinyin import chinese_to_tokens
from voxflow.text.symbols import DEFAULT_TABLE, SymbolTable

_CJK_RE = re.compile(r"[一-鿿]")


@dataclass
class FrontendOutput:
    """前端处理结果。"""

    text: str
    tokens: list[str] = field(default_factory=list)
    ids: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.ids)


def _is_cjk(ch: str) -> bool:
    return bool(_CJK_RE.match(ch))


def _char_language(ch: str) -> str | None:
    """判断单个字符的语言归属；标点 / 空格返回 ``None`` 表示"中性"。"""
    if _is_cjk(ch):
        return "zh"
    if ch.isascii() and (ch.isalpha() or ch.isdigit()):
        return "en"
    return None


def segment_by_language(text: str) -> list[tuple[str, str]]:
    """把文本切成 ``(lang, substring)`` 片段，lang ∈ {"zh", "en"}。

    中性字符（标点 / 空格）会并入当前语言片段，避免在 "你好，world"
    这类中英混排里被切碎，从而减少多余的停顿 token。
    """
    segments: list[tuple[str, str]] = []
    buf = ""
    cur: str | None = None
    for ch in text:
        lang = _char_language(ch)
        if lang is None:
            lang = cur if cur is not None else "en"
        if cur is None:
            cur, buf = lang, ch
        elif lang == cur:
            buf += ch
        else:
            segments.append((cur, buf))
            cur, buf = lang, ch
    if cur is not None:
        segments.append((cur, buf))
    return segments


class TextFrontend:
    """把文本转成音素 token 与符号表 id。"""

    def __init__(self, table: SymbolTable = DEFAULT_TABLE, add_bos_eos: bool = True) -> None:
        self.table = table
        self.add_bos_eos = add_bos_eos

    def phonemize(self, text: str, language: str = "auto") -> list[str]:
        norm = self._normalize(text, language)
        tokens: list[str] = []
        for lang, seg in segment_by_language(norm):
            tokens.extend(chinese_to_tokens(seg) if lang == "zh" else english_to_tokens(seg))
        return tokens

    def encode(self, text: str, language: str = "auto") -> FrontendOutput:
        norm = self._normalize(text, language)
        tokens: list[str] = []
        for lang, seg in segment_by_language(norm):
            tokens.extend(chinese_to_tokens(seg) if lang == "zh" else english_to_tokens(seg))
        ids = self.table.to_ids(tokens, add_bos_eos=self.add_bos_eos)
        return FrontendOutput(text=norm, tokens=tokens, ids=ids)

    __call__ = encode

    @staticmethod
    def _normalize(text: str, language: str) -> str:
        norm_lang = "zh" if language in ("zh", "auto") else language
        return normalize(text, norm_lang)


DEFAULT_FRONTEND = TextFrontend()
