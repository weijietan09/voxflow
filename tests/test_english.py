"""英文 G2P 的单元测试。"""

from voxflow.text.english import english_to_tokens, word_to_phonemes


def test_lexicon_hit():
    assert word_to_phonemes("hello") == ["HH", "AH", "L", "OW"]


def test_digraph_fallback():
    assert word_to_phonemes("ship") == ["SH", "IH", "P"]


def test_plain_letter_fallback():
    assert word_to_phonemes("cat") == ["K", "AH", "T"]


def test_sentence_with_punctuation():
    toks = english_to_tokens("hi.")
    assert toks[-1] == "<sil>"
    assert toks[:-1]  # 有音素输出


def test_digit_reads_as_word():
    assert english_to_tokens("5") == ["F", "AY", "V"]
