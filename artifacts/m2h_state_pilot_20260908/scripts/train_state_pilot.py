from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset


DEFAULT_PROJECT_ROOT = Path("/data/muxiangyu/pythonPrograms/M2HImage")
DEFAULT_DATA_ROOT = Path("/data/muxiangyu/datasets/M2HImage/M2H_Final_v2")
DEFAULT_A2_INIT = DEFAULT_DATA_ROOT / "phase1/phase2_a2_diff_r16_4000_768x1024/checkpoints/final"
DEFAULT_BASE_GLOBAL_STEP = 8400
LATENT_SHAPE = (3072, 64)
MASK_SHAPE = (3072,)
ALLOWED_INACTIVE_HAIR_MISSING = {
    "hair_gate",
    "hair_pos_proj.weight",
    "hair_proj.bias",
    "hair_proj.weight",
}


@dataclass(frozen=True)
class PilotRow:
    mid: str
    jid: str
    kid: str
    source_group: str


def add_project_root(project_root: str | Path) -> None:
    root = str(Path(project_root))
    if root not in sys.path:
        sys.path.insert(0, root)


def load_local_state_model_module():
    path = Path(__file__).with_name("state_pilot_model.py")
    spec = importlib.util.spec_from_file_location("state_pilot_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("state_pilot_model", module)
    spec.loader.exec_module(module)
    return module


def setup_dist() -> tuple[int, int, int]:
    if "RANK" not in os.environ:
        return 0, 1, int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group(backend="nccl")
    return int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])


def cleanup_dist() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def read_pool(path: str | Path) -> list[PilotRow]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise RuntimeError("--pool must contain a JSON list or an object with rows")
    parsed: list[PilotRow] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError(f"pool row {index} is not an object")
        if {"mid", "jid", "kid", "source_group"} <= set(row):
            parsed.append(PilotRow(*(str(row[key]) for key in ("mid", "jid", "kid", "source_group"))))
            continue
        refs = row.get("refs", row.get("reference_ids", row.get("ref_ids")))
        if "mid" not in row or "source_group" not in row or not isinstance(refs, list):
            raise RuntimeError(
                f"pool row {index} must be either {{mid,jid,kid,source_group}} "
                "or {mid,source_group,refs[32]}"
            )
        if len(refs) != 32:
            raise RuntimeError(f"pool row {index} has {len(refs)} refs, expected frozen 32")
        if len(set(str(value) for value in refs)) != len(refs):
            raise RuntimeError(f"pool row {index} refs contain duplicates")
        mid = str(row["mid"])
        source_group = str(row["source_group"])
        for ref_index, jid in enumerate(refs):
            kid = refs[(ref_index + 1) % len(refs)]
            parsed.append(PilotRow(mid, str(jid), str(kid), source_group))
    if not parsed:
        raise RuntimeError("--pool is empty")
    return parsed


def deterministic_pool_index(step: int, micro: int, rank: int, world_size: int, pool_size: int) -> int:
    ordinal = int(step) * max(1, int(world_size)) + int(rank)
    ordinal = ordinal * 1_000_003 + int(micro) * max(1, int(world_size)) + int(rank)
    return ordinal % int(pool_size)


def deterministic_noise_seed(base_seed: int, step: int, micro: int, rank: int) -> int:
    return (
        int(base_seed) * 1_000_003
        + int(step) * 97_409
        + int(micro) * 65_537
        + int(rank) * 31_337
    ) & ((1 << 63) - 1)


class StatePilotDataset(Dataset):
    def __init__(self, rows: list[PilotRow], cache_root: str | Path, arm: str) -> None:
        self.rows = rows
        self.cache_root = Path(cache_root)
        self.arm = arm
        self.prompt_path = self.cache_root / "prompt.npz"
        if not self.prompt_path.exists():
            raise FileNotFoundError(f"missing prompt cache: {self.prompt_path}")
        self.audit = self.audit_cache()

    def __len__(self) -> int:
        return len(self.rows)

    def _npz_path(self, prefix: str, sample_id: str) -> Path:
        return self.cache_root / prefix / f"{sample_id}.npz"

    @staticmethod
    def _load_required(path: Path, keys: Iterable[str]) -> dict[str, np.ndarray]:
        if not path.exists():
            raise FileNotFoundError(str(path))
        with np.load(path, mmap_mode="r", allow_pickle=False) as payload:
            missing = [key for key in keys if key not in payload.files]
            if missing:
                raise RuntimeError(f"{path} missing keys {missing}")
            values = {key: np.asarray(payload[key]) for key in keys}
        for key, value in values.items():
            if not np.isfinite(value).all():
                raise RuntimeError(f"{path}:{key} contains non-finite values")
        return values

    @staticmethod
    def _validate_shape(path: Path, values: dict[str, np.ndarray], key: str, shape: tuple[int, ...]) -> None:
        if tuple(values[key].shape) != shape:
            raise RuntimeError(f"{path}:{key} shape={values[key].shape}, expected {shape}")

    @staticmethod
    def _validate_dtype(path: Path, values: dict[str, np.ndarray], key: str, dtype: np.dtype) -> None:
        if values[key].dtype != dtype:
            raise RuntimeError(f"{path}:{key} dtype={values[key].dtype}, expected {dtype}")

    def _requires_teacher_endpoints(self) -> bool:
        return self.arm != "C-H"

    def audit_cache(self) -> dict[str, Any]:
        missing: list[str] = []
        bad: list[str] = []
        for row in self.rows:
            required = [
                ("M", row.mid, ("pose_latents", "garment_grid", "head_pose")),
                ("I", row.mid, ("pulid_id_embed", "appearance", "train_embed")),
                ("I", row.jid, ("pulid_id_embed", "appearance", "train_embed")),
                ("I", row.kid, ("pulid_id_embed", "appearance", "train_embed")),
                ("H", row.mid, ("target_latents", "cloth_safe_z", "body_bg_z", "face_z")),
            ]
            for prefix, sample_id, keys in required:
                path = self._npz_path(prefix, sample_id)
                try:
                    values = self._load_required(path, keys)
                    if prefix == "H":
                        self._validate_shape(path, values, "target_latents", LATENT_SHAPE)
                        for key in ("cloth_safe_z", "body_bg_z", "face_z"):
                            self._validate_shape(path, values, key, MASK_SHAPE)
                except FileNotFoundError:
                    missing.append(str(path))
                except Exception as exc:  # noqa: BLE001
                    bad.append(f"{path}: {exc}")
            if self._requires_teacher_endpoints():
                for pair in ((row.mid, row.jid), (row.mid, row.kid)):
                    path = self.cache_root / "endpoints" / f"{pair[0]}__{pair[1]}.npz"
                    try:
                        values = self._load_required(path, ("target_latents",))
                        self._validate_shape(path, values, "target_latents", LATENT_SHAPE)
                        self._validate_dtype(path, values, "target_latents", np.dtype("float32"))
                    except FileNotFoundError:
                        missing.append(str(path))
                    except Exception as exc:  # noqa: BLE001
                        bad.append(f"{path}: {exc}")
        with np.load(self.prompt_path, mmap_mode="r", allow_pickle=False) as prompt:
            prompt_missing = [key for key in ("prompt_embeds", "pooled_prompt_embeds") if key not in prompt.files]
            if prompt_missing:
                bad.append(f"{self.prompt_path}: missing keys {prompt_missing}")
        if missing or bad:
            preview = "\n".join((missing + bad)[:30])
            raise RuntimeError(
                f"state pilot cache audit failed: missing={len(missing)} bad={len(bad)}\n{preview}"
            )
        return {"rows": len(self.rows), "missing": 0, "bad": 0}

    def provenance_hashes(self) -> dict[str, Any]:
        role_hashes = {}
        audit_path = self.cache_root / "audit.json"
        if not audit_path.is_file():
            raise RuntimeError("missing fresh-cache provenance audit")
        audit = json.loads(audit_path.read_text())
        if audit.get("status") != "complete":
            raise RuntimeError("fresh cache not complete")
        for role, records, id_key in (("M", "M_roles", "mid"), ("I", "I_roles", "jid"), ("H", "H_roles", "mid")):
            for row in audit[records]:
                path = self.cache_root / role / (row[id_key] + ".npz")
                digest = sha256_file(path)
                if digest != row["cache_sha256"]:
                    raise RuntimeError("input/supervision cache changed: " + str(path))
                role_hashes[str(path)] = digest
        endpoint_hashes = {}
        if self._requires_teacher_endpoints():
            for row in self.rows:
                for jid in (row.jid, row.kid):
                    path = self.cache_root / "endpoints" / f"{row.mid}__{jid}.npz"
                    endpoint_hashes[f"{row.mid}__{jid}"] = sha256_file(path)
        return {
            "cache_root": str(self.cache_root),
            "arm": self.arm,
            "prompt_sha256": sha256_file(self.prompt_path),
            "cache_audit_sha256": sha256_file(audit_path),
            "role_cache_sha256": role_hashes,
            "endpoint_sha256": dict(sorted(endpoint_hashes.items())),
        }

    def item_for(self, index: int) -> dict[str, Any]:
        row = self.rows[int(index) % len(self.rows)]
        m = self._load_required(
            self._npz_path("M", row.mid),
            ("pose_latents", "garment_grid", "head_pose"),
        )
        h = self._load_required(
            self._npz_path("H", row.mid),
            ("target_latents", "cloth_safe_z", "body_bg_z", "face_z"),
        )
        i_mid = self._load_required(
            self._npz_path("I", row.mid),
            ("pulid_id_embed", "appearance", "train_embed"),
        )
        i_j = self._load_required(
            self._npz_path("I", row.jid),
            ("pulid_id_embed", "appearance", "train_embed"),
        )
        i_k = self._load_required(
            self._npz_path("I", row.kid),
            ("pulid_id_embed", "appearance", "train_embed"),
        )
        if self._requires_teacher_endpoints():
            endpoint_j = self._load_required(
                self.cache_root / "endpoints" / f"{row.mid}__{row.jid}.npz",
                ("target_latents",),
            )["target_latents"]
            endpoint_k = self._load_required(
                self.cache_root / "endpoints" / f"{row.mid}__{row.kid}.npz",
                ("target_latents",),
            )["target_latents"]
        else:
            endpoint_j = h["target_latents"]
            endpoint_k = h["target_latents"]
        if endpoint_j.shape != h["target_latents"].shape or endpoint_k.shape != h["target_latents"].shape:
            raise RuntimeError(
                f"teacher endpoint shape mismatch for mid={row.mid}: "
                f"H={h['target_latents'].shape} j={endpoint_j.shape} k={endpoint_k.shape}"
            )
        with np.load(self.prompt_path, mmap_mode="r", allow_pickle=False) as prompt:
            prompt_embeds = np.asarray(prompt["prompt_embeds"])
            pooled_prompt_embeds = np.asarray(prompt["pooled_prompt_embeds"])
        delta = float(np.clip(1.0 - np.dot(
            i_j["train_embed"].reshape(-1).astype(np.float32),
            i_k["train_embed"].reshape(-1).astype(np.float32),
        ), 0.0, 2.0))
        return {
            "sample_id": row.mid,
            "cf_j_id": row.jid,
            "cf_k_id": row.kid,
            "source_group": row.source_group,
            "target_latents": torch.from_numpy(h["target_latents"]).float(),
            "pose_latents": torch.from_numpy(m["pose_latents"]).float(),
            "prompt_embeds": torch.from_numpy(prompt_embeds).float(),
            "pooled_prompt_embeds": torch.from_numpy(pooled_prompt_embeds).float(),
            "pulid_id_embed": torch.from_numpy(i_mid["pulid_id_embed"]).float(),
            "appearance": torch.from_numpy(i_mid["appearance"]).float(),
            "garment": torch.from_numpy(m["garment_grid"]).float(),
            "head_pose": torch.from_numpy(m["head_pose"]).float(),
            "head_pose_is_null": torch.tensor(False, dtype=torch.float32),
            "cf_j_pulid_id_embed": torch.from_numpy(i_j["pulid_id_embed"]).float(),
            "cf_k_pulid_id_embed": torch.from_numpy(i_k["pulid_id_embed"]).float(),
            "cf_j_appearance": torch.from_numpy(i_j["appearance"]).float(),
            "cf_k_appearance": torch.from_numpy(i_k["appearance"]).float(),
            "cf_j_train_embed": torch.from_numpy(i_j["train_embed"]).float(),
            "cf_k_train_embed": torch.from_numpy(i_k["train_embed"]).float(),
            "delta_arc_jk": torch.tensor(delta, dtype=torch.float32),
            "cloth_safe_z": torch.from_numpy(h["cloth_safe_z"]).float(),
            "body_bg_z": torch.from_numpy(h["body_bg_z"]).float(),
            "face_z": torch.from_numpy(h["face_z"]).float(),
            "cf_j_state_latents": torch.from_numpy(endpoint_j).float(),
            "cf_k_state_latents": torch.from_numpy(endpoint_k).float(),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.item_for(index)


def clone_batch(batch: dict[str, Any], device: torch.device | None = None) -> dict[str, Any]:
    cloned = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            cloned[key] = value.clone() if device is None else value.clone().to(device)
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


def inject_deterministic_noise(batch: dict[str, Any], seed: int, tau_min: float = 0.0, tau_max: float = 1.0) -> dict[str, Any]:
    generator = torch.Generator(device=batch["target_latents"].device)
    generator.manual_seed(int(seed))
    batch["noise_override"] = torch.randn(
        batch["target_latents"].shape,
        generator=generator,
        device=batch["target_latents"].device,
        dtype=batch["target_latents"].dtype,
    )
    tau = torch.rand(
        (batch["target_latents"].shape[0],),
        generator=generator,
        device=batch["target_latents"].device,
        dtype=torch.float32,
    )
    batch["tau_override"] = tau_min + (tau_max - tau_min) * tau
    return batch


def monkeypatch_prepare_flow_for_noise(model: torch.nn.Module) -> None:
    original = model._prepare_flow

    def patched(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if "noise_override" not in batch:
            return original(batch)
        flow = original(batch)
        z1 = batch["noise_override"].to(device=flow["z0"].device, dtype=flow["z0"].dtype)
        tau = flow["tau"]
        z0 = flow["z0"]
        flow["z1"] = z1
        flow["z_tau"] = (1.0 - tau.view(-1, 1, 1)) * z0 + tau.view(-1, 1, 1) * z1
        flow["target_v"] = z1 - z0
        return flow

    model._prepare_flow = patched  # type: ignore[method-assign]


def atomic_torch_save_fsync(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(payload, tmp)
    with tmp.open("rb") as handle:
        os.fsync(handle.fileno())
    tmp.replace(path)
    dir_fd = os.open(str(path.parent), os.O_DIRECTORY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def torch_load_any(path: Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def checkpoint_key_is_active(name: str, model: torch.nn.Module) -> bool:
    params = dict(model.named_parameters())
    candidates = (name, f"adapter.{name}", f"transformer.{name}")
    return any(key in params and params[key].requires_grad for key in candidates)


def allowed_inactive_missing(name: str, model: torch.nn.Module) -> bool:
    if name not in ALLOWED_INACTIVE_HAIR_MISSING:
        return False
    if bool(getattr(model, "spatial_hair_enabled", False)):
        return False
    params = dict(model.named_parameters())
    candidates = (name, f"adapter.{name}", f"transformer.{name}")
    return any(key in params and not params[key].requires_grad for key in candidates)


def strict_active_checkpoint_audit(checkpoint_dir: Path, model: torch.nn.Module) -> dict[str, Any]:
    from peft import get_peft_model_state_dict

    payload = torch_load_any(checkpoint_dir / "trainable.pt", map_location="cpu")
    current_adapter = model.adapter.state_dict()
    current_lora = get_peft_model_state_dict(model.transformer)
    audit: dict[str, Any] = {
        "checkpoint": str(checkpoint_dir),
        "adapter_missing": sorted(set(current_adapter) - set(payload["adapter"])),
        "adapter_unexpected": sorted(set(payload["adapter"]) - set(current_adapter)),
        "lora_missing": sorted(set(current_lora) - set(payload["transformer_lora"])),
        "lora_unexpected": sorted(set(payload["transformer_lora"]) - set(current_lora)),
        "allowed_inactive_missing": [],
    }
    active_missing = [
        name for name in audit["adapter_missing"] + audit["lora_missing"]
        if checkpoint_key_is_active(name, model) or not allowed_inactive_missing(name, model)
    ]
    audit["active_missing"] = sorted(active_missing)
    audit["allowed_inactive_missing"] = sorted(
        set(audit["adapter_missing"] + audit["lora_missing"]) - set(active_missing)
    )
    if active_missing or audit["adapter_unexpected"] or audit["lora_unexpected"]:
        raise RuntimeError(f"strict active checkpoint audit failed: {json.dumps(audit, indent=2)}")
    for key, tensor in payload["adapter"].items():
        if key in current_adapter and tuple(tensor.shape) != tuple(current_adapter[key].shape):
            raise RuntimeError(f"adapter checkpoint shape mismatch for {key}: {tuple(tensor.shape)} vs {tuple(current_adapter[key].shape)}")
    for key, tensor in payload["transformer_lora"].items():
        if key in current_lora and tuple(tensor.shape) != tuple(current_lora[key].shape):
            raise RuntimeError(f"LoRA checkpoint shape mismatch for {key}: {tuple(tensor.shape)} vs {tuple(current_lora[key].shape)}")
    return audit


def save_pilot_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    pilot_update_step: int,
    cfg: dict[str, Any],
    rng: dict[str, Any],
    provenance: dict[str, Any],
) -> None:
    from peft import get_peft_model_state_dict

    rank = dist.get_rank() if dist.is_initialized() else 0
    rng_by_rank = None
    if dist.is_initialized():
        gathered: list[dict[str, Any] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, rng)
        rng_by_rank = {str(index): value for index, value in enumerate(gathered)}
    else:
        rng_by_rank = {"0": rng}
    if rank != 0:
        return
    payload = {
        "step": int(global_step),
        "pilot_update_step": int(pilot_update_step),
        "adapter": model.adapter.state_dict(),
        "transformer_lora": get_peft_model_state_dict(model.transformer),
        "optimizer": optimizer.state_dict(),
        "config": cfg,
        "rng_by_rank": rng_by_rank,
        "provenance": provenance,
    }
    atomic_torch_save_fsync(payload, path / "trainable.pt")
    ready = path / "READY"
    tmp = ready.with_suffix(".tmp")
    tmp.write_text(str(int(global_step)), encoding="utf-8")
    with tmp.open("rb") as handle:
        os.fsync(handle.fileno())
    tmp.replace(ready)


def load_pilot_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    base_global_step: int = DEFAULT_BASE_GLOBAL_STEP,
    rank: int = 0,
    expected_provenance: dict[str, Any] | None = None,
) -> tuple[int, int, dict[str, Any]]:
    from peft import set_peft_model_state_dict
    from peft import get_peft_model_state_dict

    payload = torch_load_any(path / "trainable.pt", map_location="cpu")
    if expected_provenance is not None and payload.get("provenance") != expected_provenance:
        raise RuntimeError("resume provenance mismatch: pool/cache/endpoint/config pin changed")
    model.adapter.load_state_dict(payload["adapter"], strict=True)
    set_peft_model_state_dict(model.transformer, payload["transformer_lora"])
    restored_lora = get_peft_model_state_dict(model.transformer)
    for key, expected in payload["transformer_lora"].items():
        actual = restored_lora.get(key)
        if actual is None or not torch.equal(actual.cpu(), expected.cpu()):
            raise RuntimeError(f"LoRA restore mismatch for {key}")
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    global_step = int(payload["step"])
    pilot_update_step = int(payload.get("pilot_update_step", global_step - int(base_global_step)))
    if global_step != int(base_global_step) + pilot_update_step:
        raise RuntimeError(
            f"checkpoint step mismatch: step={global_step}, "
            f"base={base_global_step}, pilot_update_step={pilot_update_step}"
        )
    rng_by_rank = payload.get("rng_by_rank")
    if rng_by_rank is None and "rng" in payload:
        rng_by_rank = {"0": payload["rng"]}
    rng = dict(rng_by_rank[str(rank)])
    return global_step, pilot_update_step, rng


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_a2_trainable_checkpoint(path: Path, model: torch.nn.Module) -> int:
    from peft import set_peft_model_state_dict
    from peft import get_peft_model_state_dict

    payload = torch_load_any(path / "trainable.pt", map_location="cpu")
    strict_active_checkpoint_audit(path, model)
    model.adapter.load_state_dict(payload["adapter"], strict=False)
    set_peft_model_state_dict(model.transformer, payload["transformer_lora"])
    restored_lora = get_peft_model_state_dict(model.transformer)
    for key, expected in payload["transformer_lora"].items():
        actual = restored_lora.get(key)
        if actual is None or not torch.equal(actual.cpu(), expected.cpu()):
            raise RuntimeError(f"A2 LoRA restore mismatch for {key}")
    if "differential_state" in payload and hasattr(model, "load_differential_state"):
        model.load_differential_state(payload["differential_state"])
    if payload.get("continuation_origin_step") is not None:
        model.run_origin_step = int(payload["continuation_origin_step"])
    return int(payload.get("step", 0))


def structural_checkpoint_audit(path: Path, base_global_step: int) -> dict[str, Any]:
    payload = torch_load_any(path / "trainable.pt", map_location="cpu")
    required = {
        "step",
        "pilot_update_step",
        "adapter",
        "transformer_lora",
        "optimizer",
        "config",
        "rng_by_rank",
        "provenance",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise RuntimeError(f"checkpoint missing required keys {missing}: {path}")
    step = int(payload["step"])
    updates = int(payload["pilot_update_step"])
    if step != int(base_global_step) + updates:
        raise RuntimeError(
            f"checkpoint step mismatch: step={step} base={base_global_step} updates={updates}"
        )
    ready = (path / "READY").read_text(encoding="utf-8").strip()
    if int(ready) != step:
        raise RuntimeError(f"READY step {ready} does not match checkpoint step {step}")
    if not isinstance(payload["optimizer"], dict) or "param_groups" not in payload["optimizer"]:
        raise RuntimeError("checkpoint optimizer state is not restorable")
    rng_by_rank = payload["rng_by_rank"]
    if not isinstance(rng_by_rank, dict) or not rng_by_rank:
        raise RuntimeError("checkpoint missing per-rank RNG state")
    return {
        "status": "ok",
        "checkpoint": str(path),
        "step": step,
        "pilot_update_step": updates,
        "rng_ranks": sorted(rng_by_rank),
    }


def set_pilot_train_modes(model: torch.nn.Module) -> None:
    model.train()
    for name in ("controlnet", "vae", "train_recognizer"):
        module = getattr(model, name, None)
        if module is not None:
            module.requires_grad_(False)
            module.eval()
    model.transformer.train()
    model.adapter.train()


def build_cfg(args: argparse.Namespace, world_size: int) -> dict[str, Any]:
    from conditions import load_yaml

    cfg = load_yaml(args.config)
    cfg["experiment"]["id"] = args.output_id
    cfg["experiment"]["output_root"] = str(args.output_root)
    cfg["experiment"]["seed"] = int(args.seed)
    cfg["data"]["root"] = str(args.data_root)
    cfg["model"]["lora_rank"] = int(args.lora_rank)
    cfg["model"]["load_vae_in_train"] = True
    cfg["training"]["total_steps"] = int(args.total_steps)
    cfg["training"]["checkpoint_every"] = int(args.checkpoint_every)
    cfg["training"]["benchmark_steps"] = int(args.health_steps)
    cfg["training"]["baseline_global_batch"] = int(args.global_batch)
    cfg["training"]["baseline_lr"] = float(args.lr)
    cfg["training"]["micro_batch"] = int(args.micro_batch)
    cfg["training"]["grad_accum"] = int(args.grad_accum) if args.grad_accum else math.ceil(
        int(args.global_batch) / (world_size * int(args.micro_batch))
    )
    cfg["training"]["optimizer"] = str(args.optimizer)
    diff = cfg["training"].setdefault("differential", {})
    diff.update(
        {
            "enabled": True,
            "lambda_teach": 0.5,
            "lambda_inv": 0.2,
            "lambda_hinge": 0.0,
            "hinge_g": None,
            "hinge_g_resolved": None,
            "tau_min": 0.2,
            "tau_max": 0.8,
            "diff_every": 1,
        }
    )
    diff.setdefault("decode", {})
    diff["decode"].update({"enabled": True, "freq": 3})
    diff.setdefault("identity_loss", {})
    diff["identity_loss"].update(
        {
            "enabled": True,
            "lambda_dir": 0.1,
            "lambda_abs": 0.05,
            "max_skip_rate": 0.5,
            "compensate_grad_accum": True,
        }
    )
    cfg["_runtime"] = {
        "world_size": world_size,
        "micro_batch": int(args.micro_batch),
        "grad_accum": int(cfg["training"]["grad_accum"]),
        "global_batch": world_size * int(args.micro_batch) * int(cfg["training"]["grad_accum"]),
        "effective_lr": float(args.lr),
        "state_pilot_arm": args.arm,
        "ddp_find_unused_parameters": False,
        "base_global_step": int(args.base_global_step),
        "max_gpu_hours": float(args.max_gpu_hours),
    }
    if cfg["_runtime"]["global_batch"] != int(args.global_batch):
        raise RuntimeError(f"global batch resolved to {cfg['_runtime']['global_batch']}, expected {args.global_batch}")
    return cfg


def rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def collate_single(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 1:
        raise RuntimeError("state pilot runner expects micro_batch=1")
    row = rows[0]
    out: dict[str, Any] = {}
    for key, value in row.items():
        out[key] = value.unsqueeze(0) if isinstance(value, torch.Tensor) else [value]
    return out


def build_model_for_checkpoint(args: argparse.Namespace, cfg: dict[str, Any], device: torch.device, dtype: torch.dtype):
    state_module = load_local_state_model_module()
    from train_paired import load_components, load_directed_identity_components

    transformer, controlnet, vae, adapter, pulid, load_notes = load_components(cfg, device, dtype)
    recognizer, detector, identity_notes = load_directed_identity_components(cfg, device)
    load_notes.update(identity_notes)
    model = state_module.StatePilotDirectedFlowModel(
        transformer,
        controlnet,
        adapter,
        pulid,
        vae,
        recognizer,
        detector,
        cfg,
        pilot_arm=args.arm,
    )
    monkeypatch_prepare_flow_for_noise(model)
    return model, load_notes


def verify_checkpoint_only(args: argparse.Namespace) -> None:
    print(
        json.dumps(
            structural_checkpoint_audit(
                Path(args.verify_checkpoint),
                int(args.base_global_step),
            ),
            indent=2,
        ),
        flush=True,
    )


def train(args: argparse.Namespace) -> None:
    if args.pool is None or args.cache_root is None:
        raise RuntimeError("--pool and --cache-root are required for training mode")
    add_project_root(args.project_root)
    from conditions import choose_dtype, save_yaml, seed_everything
    from train_paired import (
        assert_gate_optimizer_lr,
        build_optimizer,
        reduce_sparse_metric_sums,
        averaged_metrics,
        accumulate_scalar_metrics,
    )

    rank, world_size, local_rank = setup_dist()
    if world_size != int(args.expected_world_size):
        raise RuntimeError(f"state pilot expected world_size={args.expected_world_size}, got {world_size}")
    if int(args.micro_batch) != 1:
        raise RuntimeError("authorized state pilot interface requires --micro-batch 1")
    cfg = build_cfg(args, world_size)
    base_global_step = int(args.base_global_step)
    seed_everything(int(args.seed) + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(local_rank)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    dtype = choose_dtype(cfg["model"]["precision"])
    rows = read_pool(args.pool)
    dataset = StatePilotDataset(rows, args.cache_root, args.arm)
    provenance = {
        "pool": str(args.pool),
        "pool_sha256": sha256_file(Path(args.pool)),
        "config": str(args.config),
        "config_sha256": sha256_file(Path(args.config)),
        "a2_init": str(args.a2_init),
        "a2_trainable_sha256": sha256_file(Path(args.a2_init) / "trainable.pt"),
        **dataset.provenance_hashes(),
    }

    model, load_notes = build_model_for_checkpoint(args, cfg, device, dtype)
    set_pilot_train_modes(model)
    audit = strict_active_checkpoint_audit(Path(args.a2_init), model)
    a2_step = load_a2_trainable_checkpoint(Path(args.a2_init), model)
    if a2_step != base_global_step:
        raise RuntimeError(f"A2 initialization step changed: {a2_step} != {base_global_step}")
    set_pilot_train_modes(model)
    ddp = DDP(model, device_ids=[local_rank], find_unused_parameters=False) if world_size > 1 and device.type == "cuda" else model
    optimizer = build_optimizer(ddp, cfg)
    assert_gate_optimizer_lr(optimizer, cfg)

    output = Path(args.output_root) / args.output_id / args.arm
    ckpt_dir = output / "checkpoints"
    log_dir = output / "logs"
    start_update_step = 0
    if args.resume:
        _resume_global_step, start_update_step, restored_rng = load_pilot_checkpoint(
            Path(args.resume),
            model,
            optimizer,
            base_global_step=base_global_step,
            rank=rank,
            expected_provenance=provenance,
        )
        restore_rng_state(restored_rng)
    target_update_step = (
        start_update_step + int(args.health_steps)
        if args.health
        else int(args.total_steps)
    )
    accum = int(cfg["_runtime"]["grad_accum"])
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        save_yaml(output / "resolved_config.yaml", cfg)
        (output / "launch.json").write_text(json.dumps({
            "arm": args.arm,
            "pool": str(args.pool),
            "cache_root": str(args.cache_root),
            "a2_init": str(args.a2_init),
            "a2_trainable_sha256": sha256_file(Path(args.a2_init) / "trainable.pt"),
            "checkpoint_audit": audit,
            "provenance": provenance,
            "load_notes": load_notes,
            "pool_rows": len(rows),
            "source_rows": len({row.source_group for row in rows}),
            "base_global_step": base_global_step,
            "start_update_step": start_update_step,
            "target_update_step": target_update_step,
            "runtime": cfg["_runtime"],
            "risks": [
                "teacher endpoints are fixed external A4 products, not ground-truth labels",
                "pilot pool repeats by deterministic index and is not an independent confirmation set",
                "no GPU job is launched by this script until invoked under torchrun",
            ],
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    update_step = int(start_update_step)
    metric_totals: dict[str, float] = {}
    metric_counts: dict[str, int] = {}
    skip_total = 0.0
    attempt_total = 0.0
    benchmark_start = time.perf_counter()
    stop_after_checkpoint = False
    while update_step < target_update_step:
        elapsed_hours = (time.perf_counter() - benchmark_start) / 3600.0
        consumed_gpu_hours = elapsed_hours * max(1, world_size)
        if consumed_gpu_hours >= float(args.max_gpu_hours):
            raise RuntimeError(
                f"max GPU-hour budget exceeded before step: used={consumed_gpu_hours:.4f}, cap={args.max_gpu_hours}"
            )
        optimizer.zero_grad(set_to_none=True)
        for micro in range(accum):
            pool_index = deterministic_pool_index(update_step, micro, rank, world_size, len(dataset))
            batch = dataset.item_for(pool_index)
            batch = collate_single([batch])
            batch = clone_batch(batch, device)
            inject_deterministic_noise(
                batch,
                deterministic_noise_seed(int(args.seed), update_step, micro, rank),
            )
            optimizer_boundary = micro == accum - 1
            run_step = update_step
            decode_trigger = bool(run_step % 3 == 0 and optimizer_boundary)
            loss, metrics = ddp(
                batch,
                train_step=run_step,
                decode_trigger=decode_trigger,
                identity_loss_accum_scale=float(accum) if decode_trigger else 1.0,
                optimizer_boundary=optimizer_boundary,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at update_step={update_step} micro={micro} rank={rank}")
            accumulate_scalar_metrics(metric_totals, metric_counts, metrics)
            (loss / accum).backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in ddp.parameters() if p.requires_grad],
            float(cfg["training"]["max_grad_norm"]),
            error_if_nonfinite=True,
        )
        optimizer.step()
        update_step += 1
        global_step = base_global_step + update_step
        sparse = reduce_sparse_metric_sums(metric_totals, device) if device.type == "cuda" else {}
        attempt_total += float(sparse.get("id_loss_attempt_count", 0.0))
        skip_total += float(sparse.get("id_loss_skip_count", 0.0))
        if attempt_total > 0.0 and skip_total / attempt_total > 0.5:
            raise RuntimeError(f"identity skip gate exceeded 50%: {skip_total}/{attempt_total}")
        step_metrics = averaged_metrics(metric_totals, metric_counts)
        metric_totals.clear()
        metric_counts.clear()
        if rank == 0:
            elapsed = time.perf_counter() - benchmark_start
            row = {
                "step": global_step,
                "pilot_update_step": update_step,
                "arm": args.arm,
                "seconds_per_optimizer_step": elapsed / max(1, update_step - start_update_step),
                "gpu_hours_used": elapsed / 3600.0 * max(1, world_size),
                "max_gpu_hours": float(args.max_gpu_hours),
                "budget_cap_steps": target_update_step,
                "identity_skip_total": skip_total,
                "identity_attempt_total": attempt_total,
                **step_metrics,
            }
            with (log_dir / "train.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if args.health and update_step - start_update_step == int(args.health_steps):
                if row["seconds_per_optimizer_step"] > float(args.health_step_seconds_gate):
                    stop_after_checkpoint = True
        if dist.is_initialized():
            stop_flag = torch.tensor(
                [1 if stop_after_checkpoint else 0],
                device=device,
                dtype=torch.int32,
            )
            dist.broadcast(stop_flag, src=0)
            stop_after_checkpoint = bool(stop_flag.item())
        if update_step % int(args.checkpoint_every) == 0 or update_step == target_update_step or stop_after_checkpoint:
            save_pilot_checkpoint(
                ckpt_dir / f"step-{global_step:06d}",
                model,
                optimizer,
                global_step,
                update_step,
                cfg,
                rng_state(),
                provenance,
            )
        if stop_after_checkpoint:
            if rank == 0:
                (output / "STOP_HEALTH_SLOW").write_text(
                    json.dumps(
                        {
                            "reason": "health_step_seconds_gate",
                            "step": global_step,
                            "pilot_update_step": update_step,
                            "gate_seconds": float(args.health_step_seconds_gate),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            break
    final_path = ckpt_dir / "final"
    final_global_step = base_global_step + update_step
    save_pilot_checkpoint(
        final_path,
        model,
        optimizer,
        final_global_step,
        update_step,
        cfg,
        rng_state(),
        provenance,
    )
    if rank == 0:
        if args.verify_restore:
            (log_dir / "restore_audit.json").write_text(
                json.dumps(
                    structural_checkpoint_audit(final_path, int(args.base_global_step)),
                    indent=2,
                ),
                encoding="utf-8",
            )
        (output / "training_status.json").write_text(json.dumps({
            "status": "stopped_health_slow" if stop_after_checkpoint else "complete",
            "step": final_global_step,
            "pilot_update_step": update_step,
            "target_update_step": target_update_step,
            "identity_skip_total": skip_total,
            "identity_attempt_total": attempt_total,
        }, indent=2), encoding="utf-8")
    cleanup_dist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Authorized M2H M/I/H state-pilot three-arm runner.")
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--config", default=str(DEFAULT_PROJECT_ROOT / "configs/a4_directed.yaml"))
    parser.add_argument("--pool", help="JSON rows: [{mid,jid,kid,source_group}, ...]")
    parser.add_argument("--cache-root", help="Cache root with M/, I/, H/, endpoints/, prompt.npz")
    parser.add_argument("--arm", choices=("C-H", "C-perm", "E-match"), required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_DATA_ROOT / "phase1/m2h_state_pilot_20260908")
    parser.add_argument("--output-id", default="state_pilot_seed0")
    parser.add_argument("--a2-init", type=Path, default=DEFAULT_A2_INIT)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-global-step", type=int, default=DEFAULT_BASE_GLOBAL_STEP)
    parser.add_argument("--expected-world-size", type=int, default=3)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=6)
    parser.add_argument("--global-batch", type=int, default=18)
    parser.add_argument("--lr", type=float, default=5.625e-5)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--total-steps", type=int, default=300)
    parser.add_argument("--health-steps", type=int, default=20)
    parser.add_argument("--health-step-seconds-gate", type=float, default=120.0)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--max-gpu-hours", type=float, default=60.0)
    parser.add_argument("--optimizer", default="paged_adamw8bit")
    parser.add_argument("--health", action="store_true", help="Run only health_steps from current start/resume step.")
    parser.add_argument("--verify-restore", action="store_true", help="Reload final pilot checkpoint in-process after save.")
    parser.add_argument("--verify-checkpoint", type=Path, help="Fresh-process checkpoint restore verification mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verify_checkpoint:
        verify_checkpoint_only(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
