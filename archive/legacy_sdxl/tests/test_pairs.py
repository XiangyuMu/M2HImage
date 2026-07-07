from mcic.data.pairs import discover_pairs


def test_discovers_relative_same_stem_pairs(tmp_path):
    (tmp_path / "mannequin" / "look").mkdir(parents=True)
    (tmp_path / "human" / "look").mkdir(parents=True)
    (tmp_path / "mannequin" / "look" / "0001.jpg").write_bytes(b"")
    (tmp_path / "human" / "look" / "0001.png").write_bytes(b"")
    (tmp_path / "human" / "0002.jpg").write_bytes(b"")
    pairs, failures = discover_pairs(tmp_path, "mannequin", "human")
    assert [pair.sample_id for pair in pairs] == ["look/0001"]
    assert failures["missing_mannequin"] == ["0002"]
