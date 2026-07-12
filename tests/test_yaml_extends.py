from __future__ import annotations

from conditions import load_yaml


def test_yaml_extends_deep_merges(tmp_path) -> None:
    (tmp_path / 'base.yaml').write_text(
        'model:\n  rank: 16\n  nested:\n    keep: true\n    change: 1\n',
        encoding='utf-8',
    )
    (tmp_path / 'child.yaml').write_text(
        'extends: base.yaml\nmodel:\n  nested:\n    change: 2\ntraining:\n  enabled: true\n',
        encoding='utf-8',
    )
    payload = load_yaml(tmp_path / 'child.yaml')
    assert payload == {
        'model': {'rank': 16, 'nested': {'keep': True, 'change': 2}},
        'training': {'enabled': True},
    }
