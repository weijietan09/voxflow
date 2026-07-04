"""双语前端的端到端测试。"""

from voxflow.text.frontend import TextFrontend, segment_by_language
from voxflow.text.symbols import DEFAULT_TABLE


def test_segment_splits_zh_en():
    segs = segment_by_language("你好world")
    assert segs == [("zh", "你好"), ("en", "world")]


def test_punctuation_stays_with_previous_segment():
    segs = segment_by_language("你好，world")
    assert segs == [("zh", "你好，"), ("en", "world")]


def test_encode_wraps_bos_eos():
    fe = TextFrontend(add_bos_eos=True)
    out = fe.encode("你好")
    assert out.ids[0] == DEFAULT_TABLE.bos_id
    assert out.ids[-1] == DEFAULT_TABLE.eos_id


def test_encode_ids_in_range():
    fe = TextFrontend()
    out = fe.encode("你好 hello 12")
    assert len(out.ids) == len(out.tokens) + 2
    assert all(0 <= i < len(DEFAULT_TABLE) for i in out.ids)


def test_phonemize_mixed_language():
    fe = TextFrontend()
    toks = fe.phonemize("中文hello")
    assert "zh" in toks  # 声母
    assert "HH" in toks  # 英文音素


def test_callable_alias():
    fe = TextFrontend()
    assert fe("你好").tokens == fe.encode("你好").tokens
