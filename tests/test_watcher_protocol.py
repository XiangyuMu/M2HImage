from __future__ import annotations

from collections import Counter

from conditions import load_yaml
from watcher_protocol import load_watcher_eval_set, watcher_eval_set_from_config


def test_frozen_watcher_protocol_is_balanced_and_visible_hair() -> None:
    protocol = load_watcher_eval_set("configs/watcher_eval_set.json")
    assert len(protocol["sample_ids"]) == 16
    assert Counter(
        row["garment_type"] for row in protocol["samples"]
    ) == {"top": 4, "dress": 4, "pants": 4, "skirt": 4}
    assert all(row["hair_fraction"] >= 0.01 for row in protocol["samples"])


def test_resume_config_resolves_the_same_frozen_protocol() -> None:
    cfg = load_yaml("configs/spatial_warmup_resume_hair_incontext.yaml")
    protocol = watcher_eval_set_from_config(cfg)
    assert protocol["protocol_hash"] == (
        "40826872040bb9738a5b5f3b4e06b527f6d3e6a8e74d24db92326a144ec48517"
    )
