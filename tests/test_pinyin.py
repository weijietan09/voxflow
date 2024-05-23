"""汉语拼音 G2P 的单元测试。"""

from voxflow.text.pinyin import chinese_to_tokens, split_initial_final


def test_split_initial_final_two_char():
    assert split_initial_final("zhong") == ("zh", "ong")


def test_split_initial_final_single():
    assert split_initial_final("hao") == ("h", "ao")


def test_split_zero_initial():
    assert split_initial_final("ao") == ("", "ao")


def test_nihao_tokens():
    assert chinese_to_tokens("你好") == ["n", "i", "3", "h", "ao", "3"]


def test_zhongguo_tokens():
    assert chinese_to_tokens("中国") == ["zh", "ong", "1", "g", "uo", "2"]


def test_neutral_tone_is_five():
    assert chinese_to_tokens("的") == ["d", "e", "5"]


def test_punctuation_becomes_pause():
    toks = chinese_to_tokens("好。")
    assert toks[-1] == "<sil>"
