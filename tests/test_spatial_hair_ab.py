from __future__ import annotations

from conditions import load_yaml


def test_part_b_hair_ab_fairness_fields() -> None:
    space = load_yaml("configs/spatial_hair_ab_space.yaml")
    semantic = load_yaml("configs/spatial_hair_ab_semantic.yaml")
    for key in ("seed",):
        assert space["experiment"][key] == semantic["experiment"][key]
    for key in (
        "total_steps",
        "resume",
        "baseline_lr",
        "baseline_global_batch",
        "micro_batch",
        "grad_accum",
    ):
        assert space["training"][key] == semantic["training"][key]
    assert space["training"]["total_steps"] == 8400
    assert space["model"]["spatial_conditions"]["hair"]["reference"]["enabled"]
    assert not semantic["model"]["spatial_conditions"]["hair"]["reference"]["enabled"]
    assert semantic["model"]["identity_adapter"]["use_hair_tokens"]
    assert (
        semantic["model"]["identity_adapter"]["hair_token_mode"]
        == "semantic_dense"
    )
    assert semantic["model"]["identity_adapter"]["hair_semantic_max_tokens"] == 32


def test_part_c_pairs_share_start_and_steps() -> None:
    for winner in ("space", "semantic"):
        directed = load_yaml(f"configs/spatial_c_a4prime_{winner}.yaml")
        control = load_yaml(f"configs/spatial_c_cont_{winner}.yaml")
        assert directed["training"]["resume"] == control["training"]["resume"]
        assert directed["training"]["total_steps"] == 12400
        assert control["training"]["total_steps"] == 12400
        assert directed["training"]["differential"]["enabled"]
        assert directed["training"]["differential"]["sampling"] == "semihard"
        assert not control["training"]["differential"]["enabled"]
        assert directed["training"]["hair_loss"] == control["training"]["hair_loss"]
        assert directed["experiment"]["seed"] == control["experiment"]["seed"]


def test_metrics_config_registers_every_new_run() -> None:
    cfg = load_yaml("configs/spatial_hair_ab_metrics_v2.yaml")
    runs = cfg["metrics_v2"]["runs"]
    expected = {
        "spatial_hair_ab_space",
        "spatial_hair_ab_semantic",
        "spatial_c_a4prime_space",
        "spatial_c_cont_space",
        "spatial_c_a4prime_semantic",
        "spatial_c_cont_semantic",
    }
    assert expected <= set(runs)
