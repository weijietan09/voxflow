"""音素 / 符号表。

符号集合刻意做成中英文共享的一套：
- 特殊符号（pad / bos / eos / unk / 停顿）；
- 标点；
- 汉语拼音的声母、韵母与声调（声调作为独立 token）；
- 英文 ARPAbet 音素子集。

这样中英混读时能落在同一个 embedding 空间里，声学模型只需要一个符号表。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
SIL = "<sil>"  # 句间 / 词间停顿
SPECIALS = [PAD, BOS, EOS, UNK, SIL]

# 归一化后可能保留的标点（会被映射为停顿或语气符号）
PUNCTUATION = list("，。！？、；：…—,.!?;:")

# 汉语拼音声母
PINYIN_INITIALS = "b p m f d t n l g k h j q x zh ch sh r z c s y w".split()

# 汉语拼音韵母（不含声调，声调另作 token）
PINYIN_FINALS = (
    "a o e i u v ai ei ui ao ou iu ie ve er an en in un vn "
    "ang eng ing ong ia iao ian iang iong ua uo uai uan uang uen ueng io"
).split()

# 声调：1-4 声，5 表示轻声
TONES = ["1", "2", "3", "4", "5"]

# 英文 ARPAbet 音素（去掉重音数字，保留基本音素）
ARPABET = (
    "AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG "
    "OW OY P R S SH T TH UH UW V W Y Z ZH"
).split()


def build_default_symbols() -> list[str]:
    """按固定顺序拼出默认符号表（顺序决定 id，改动会影响已存权重）。"""
    symbols: list[str] = []
    for group in (SPECIALS, PUNCTUATION, PINYIN_INITIALS, PINYIN_FINALS, TONES, ARPABET):
        for s in group:
            if s not in symbols:
                symbols.append(s)
    return symbols


class SymbolTable:
    """符号与整数 id 之间的双向映射。"""

    def __init__(self, symbols: Sequence[str]) -> None:
        self.symbols = list(symbols)
        self._s2i = {s: i for i, s in enumerate(self.symbols)}
        for tok in (PAD, BOS, EOS, UNK):
            if tok not in self._s2i:
                raise ValueError(f"符号表缺少必需的特殊符号: {tok}")

    def __len__(self) -> int:
        return len(self.symbols)

    def __contains__(self, symbol: object) -> bool:
        return symbol in self._s2i

    @property
    def pad_id(self) -> int:
        return self._s2i[PAD]

    @property
    def bos_id(self) -> int:
        return self._s2i[BOS]

    @property
    def eos_id(self) -> int:
        return self._s2i[EOS]

    @property
    def unk_id(self) -> int:
        return self._s2i[UNK]

    def to_id(self, symbol: str) -> int:
        return self._s2i.get(symbol, self.unk_id)

    def to_ids(self, tokens: Iterable[str], add_bos_eos: bool = False) -> list[int]:
        ids = [self.to_id(t) for t in tokens]
        if add_bos_eos:
            ids = [self.bos_id, *ids, self.eos_id]
        return ids

    def to_tokens(self, ids: Iterable[int]) -> list[str]:
        out = []
        for i in ids:
            if 0 <= i < len(self.symbols):
                out.append(self.symbols[i])
            else:
                out.append(UNK)
        return out


DEFAULT_SYMBOLS = build_default_symbols()
DEFAULT_TABLE = SymbolTable(DEFAULT_SYMBOLS)
