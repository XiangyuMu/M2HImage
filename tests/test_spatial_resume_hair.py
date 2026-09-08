from __future__ import annotations

import torch

from conditions import load_yaml
from hair_supervision import DifferentiableHairDinoLoss
from probe_response_track import trajectory_decision
from train_paired import WarmupFlowModel


def bare_flow_model() -> WarmupFlowModel:
    model = WarmupFlowModel.__new__(WarmupFlowModel)
    torch.nn.Module.__init__(model)
    model.pair_region_weighting_enabled = True
    model.pair_region_cfg = {"w_cloth": 2.0, "w_hair": 2.0}
    return model


def test_paired_region_weighting_is_mean_normalized() -> None:
    model = bare_flow_model()
    pred = torch.zeros((1, 4, 1), dtype=torch.float32)
    target = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    batch = {
        "cloth_safe_z": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "hair_z": torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
        "face_z": torch.tensor([[0.0, 1.0, 1.0, 0.0]]),
    }
    loss, metrics = model._paired_flow_loss(pred, target, batch)
    raw_weight = torch.tensor([2.0, 2.0, 1.0, 1.0])
    normalized = raw_weight / raw_weight.mean()
    expected = (target.square().flatten() * normalized).mean()
    assert torch.allclose(loss, expected)
    assert torch.allclose(metrics["pair_weight_mean"], torch.tensor(1.0))
    assert set(("mse_cloth_safe", "mse_hair", "mse_face", "mse_other")) <= set(metrics)


def test_paired_region_weighting_can_restore_face_weight() -> None:
    model = bare_flow_model()
    model.pair_region_cfg["w_face"] = 2.0
    pred = torch.zeros((1, 4, 1), dtype=torch.float32)
    target = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    batch = {
        "cloth_safe_z": torch.zeros((1, 4)),
        "hair_z": torch.zeros((1, 4)),
        "face_z": torch.tensor([[0.0, 1.0, 1.0, 0.0]]),
    }
    loss, _ = model._paired_flow_loss(pred, target, batch)
    weights = torch.tensor([1.0, 2.0, 2.0, 1.0])
    expected = (target.square().flatten() * (weights / weights.mean())).mean()
    assert torch.allclose(loss, expected)


def test_projected_hair_mask_wiring_fails_fast() -> None:
    supervisor = DifferentiableHairDinoLoss.__new__(DifferentiableHairDinoLoss)
    torch.nn.Module.__init__(supervisor)
    supervisor.mask_source = "target_projection"
    decoded = torch.zeros((1, 3, 128, 96))
    target = torch.zeros((1, 4, 8))
    valid = torch.ones((1, 4))
    try:
        supervisor(decoded, target, valid, projected_hair_mask=None)
    except RuntimeError as exc:
        assert "projected_hair_mask" in str(exc)
    else:
        raise AssertionError("missing projected hair mask must not be classified as parser failure")


def test_projected_hair_mask_recovers_48_by_64_grid() -> None:
    decoded = torch.zeros((1, 3, 128, 96))
    token_mask = torch.zeros((1, 3072))
    token_mask[:, :48] = 1.0
    mask = DifferentiableHairDinoLoss._projected_mask(decoded, token_mask)
    assert mask.shape == (1, 1, 128, 96)
    assert float(mask.min()) >= 0.0
    assert float(mask.max()) <= 1.0


def track_row(
    step: int,
    concentration: float,
    hair: float,
    garment: float,
    hair_dino: float,
    hair_lab: float,
) -> dict:
    return {
        "step": step,
        "cloth_concentration": concentration,
        "hair_relative_response": hair,
        "hair_ref_swap_concentration": concentration,
        "garment_dino": garment,
        "garment_dino_by_type": {
            "top": garment,
            "dress": garment,
            "pants": garment,
            "skirt": garment,
        },
        "hair_dino": hair_dino,
        "hair_lab_distance": hair_lab,
        "head5_distance": 0.012,
        "body_distance": 0.015,
        "face_detection_rate": 1.0,
        "swap_cosine_max": 0.25,
    }


def test_trajectory_ignores_cloth_proxy_and_uses_hair_hard_gate() -> None:
    cfg = load_yaml("configs/spatial_warmup_resume_hair_incontext.yaml")
    rising = [
        track_row(2000, 1.00, 0.64, 0.916, 0.766, 31.9),
        track_row(2500, 1.20, 0.70, 0.918, 0.79, 29.0),
    ]
    approved = trajectory_decision(cfg, rising)
    assert approved["status"] == "approved_trend"
    assert approved["auto_approval"] is True
    assert "cloth_concentration_rising" not in approved["checks"]

    plateau = [
        track_row(2000, 1.00, 0.64, 0.916, 0.7660, 31.9),
        track_row(2500, 1.20, 0.70, 0.918, 0.7661, 31.9),
        track_row(3000, 1.21, 0.71, 0.919, 0.7661, 31.9),
    ]
    stopped = trajectory_decision(cfg, plateau)
    assert stopped["status"] == "hair_plateau"
    assert stopped["stop"] is True


def test_untrained_step2000_is_recorded_without_firing_post_train_guards() -> None:
    cfg = load_yaml("configs/spatial_warmup_resume_hair_incontext.yaml")
    baseline = [
        {
            **track_row(2000, 1.06, 0.71, 0.772, 0.597, 36.9),
            "head5_distance": 0.0248,
        }
    ]
    decision = trajectory_decision(cfg, baseline)
    assert decision["status"] == "baseline_recorded"
    assert decision["stop"] is False
    assert "garment_result_guard" in decision["baseline_guard_observation"]
    assert "head5_guard" in decision["baseline_guard_observation"]


def test_fixed_median_guard_requires_two_consecutive_regressions() -> None:
    cfg = load_yaml("configs/spatial_warmup_resume_hair_incontext.yaml")
    baseline = track_row(2000, 1.1, 0.65, 0.90, 0.70, 30.0)
    first_drop = track_row(2500, 1.1, 0.68, 0.87, 0.72, 29.0)
    warning = trajectory_decision(cfg, [baseline, first_drop])
    assert warning["stop"] is False
    assert warning["single_point_warnings"]

    second_drop = track_row(3000, 1.1, 0.70, 0.86, 0.74, 28.0)
    stopped = trajectory_decision(cfg, [baseline, first_drop, second_drop])
    assert stopped["stop"] is True
    assert "garment_result_guard" in stopped["plateau_reasons"]


def test_quality_repair_absolute_guards_need_two_failed_checkpoints() -> None:
    cfg = load_yaml("configs/spatial_quality_repair_resume.yaml")
    assert cfg["training"]["hair_loss"]["dino_precision"] == "bf16"
    baseline = {
        **track_row(4400, 1.1, 0.7, 0.755753, 0.76, 31.0),
        "body_distance": 0.012192,
        "head5_distance": 0.023175,
        "face_detection_rate": 1.0,
        "swap_cosine_max": 0.25,
    }
    failed_once = {
        **track_row(5000, 1.1, 0.71, 0.72, 0.77, 30.0),
        "body_distance": 0.020,
        "head5_distance": 0.030,
        "face_detection_rate": 0.96,
        "swap_cosine_max": 0.40,
    }
    warning = trajectory_decision(cfg, [baseline, failed_once])
    assert warning["stop"] is False
    assert warning["single_point_warnings"]

    failed_twice = {**failed_once, "step": 5500}
    stopped = trajectory_decision(cfg, [baseline, failed_once, failed_twice])
    assert stopped["stop"] is True
    assert {
        "garment_result_guard",
        "head5_guard",
        "body_guard",
        "face_detection_guard",
        "identity_swap_guard",
    } <= set(stopped["plateau_reasons"])


def test_resume_config_records_protocol() -> None:
    cfg = load_yaml("configs/spatial_warmup_resume_hair_incontext.yaml")
    assert cfg["training"]["resume"].endswith("step-002500")
    assert "fixedset" in cfg["experiment"]["id"]
    assert cfg["eval"]["watcher_eval_set"] == (
        "configs/watcher_eval_set.json"
    )
    assert cfg["training"]["paired_region_weighting"] == {
        "enabled": True,
        "w_cloth": 2.0,
        "w_hair": 2.0,
    }
    assert cfg["training"]["hair_loss"]["lambda_hair"] == 0.5
    assert cfg["training"]["hair_loss"]["decode_freq"] == 2
    assert cfg["model"]["spatial_conditions"]["hair"]["reference"]["enabled"] is True
    assert cfg["model"]["spatial_conditions"]["hair"]["reference"]["stride"] == 2
    assert cfg["model"]["identity_adapter"]["use_hair_tokens"] is False
    assert cfg["model"]["identity_adapter"]["retain_legacy_hair_parameters_for_resume"] is True
    assert cfg["model"]["identity_adapter"]["gate_lr_mult"] == 10.0
    assert cfg["eval"]["manual_review_step"] == 0
    track = cfg["eval"]["response_track"]
    assert track["visible_hair_sample_count"] == 16
    assert track["baseline_snapshot"] == (
        "artifacts/rebaseline_fixed_set/step2000.json"
    )
