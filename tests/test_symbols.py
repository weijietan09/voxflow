"""符号表的单元测试。"""

from voxflow.text.symbols import DEFAULT_TABLE, build_default_symbols


def test_specials_present():
    for tok in ("<pad>", "<bos>", "<eos>", "<unk>", "<sil>"):
        assert tok in DEFAULT_TABLE


def test_ids_are_unique_and_dense():
    symbols = build_default_symbols()
    assert len(symbols) == len(set(symbols))
    assert DEFAULT_TABLE.pad_id == 0


def test_roundtrip_tokens():
    ids = DEFAULT_TABLE.to_ids(["zh", "ong", "1"])
    assert DEFAULT_TABLE.to_tokens(ids) == ["zh", "ong", "1"]


def test_unknown_symbol_maps_to_unk():
    assert DEFAULT_TABLE.to_id("绝不存在的符号") == DEFAULT_TABLE.unk_id
