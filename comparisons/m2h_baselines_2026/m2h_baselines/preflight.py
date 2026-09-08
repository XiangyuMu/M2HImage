from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import yaml
from PIL import Image


@contextmanager
def repo_import(repo: Path) -> Iterator[None]:
    original_cwd = Path.cwd()
    original_path = list(sys.path)
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    try:
        yield
    finally:
        os.chdir(original_cwd)
        sys.path[:] = original_path


def _shape(value: object) -> list[int]:
    shape = getattr(value, "shape", None)
    return list(shape) if shape is not None else []


def _probe_refton(repo: Path, layout: Path, split: str) -> dict[str, object]:
    with repo_import(repo):
        module = importlib.import_module("datasets_util.viton")
        dataset = module.VITONDataset(
            layout,
            size=(512, 512),
            train=split == "train",
            center_crop=True,
        )
        sample = dataset[0]
    required = {"image", "person", "agnostic", "cloth", "image_ref", "dense"}
    if missing := sorted(required - set(sample)):
        raise KeyError(f"RefTon sample missing keys: {missing}")
    if np.array_equal(sample["image"].numpy(), sample["person"].numpy()):
        raise ValueError("RefTon target and mannequin person tensors are identical")
    return {"length": len(dataset), "keys": sorted(sample), "target": _shape(sample["image"])}


def _probe_ita(repo: Path, layout: Path, split: str) -> dict[str, object]:
    with repo_import(repo):
        module = importlib.import_module("scr.dataset_vitonhd")
        data_root = layout / "zalando-hd-resized"
        dataset = module.VITONHDDataset(
            data_root_dir=str(data_root),
            img_H=512,
            img_W=512,
            vit_img_H=224,
            vit_img_W=224,
            is_paired=True,
            is_test=split != "train",
            is_sorted=True,
        )
        sample = dataset[0]
    mask_name = f"{Path(dataset.im_names[0]).stem}_mask.png"
    disk_mask = np.asarray(
        Image.open(data_root / dataset.data_type / "agnostic-mask" / mask_name).convert("L"),
        dtype=np.float32,
    )
    replace_fraction = float((disk_mask >= 128).mean())
    loader_keep_fraction = float(np.asarray(sample["agn_mask"])[0].mean())
    if "replace_mask" not in sample:
        raise KeyError("ITA sample missing explicit replace_mask loss tensor")
    loader_replace = np.asarray(sample["replace_mask"], dtype=np.float32)[0]
    if not np.isin(loader_replace, (0.0, 1.0)).all():
        raise ValueError("ITA replace_mask loss tensor is not binary")
    loader_replace_fraction = float(loader_replace.mean())
    if abs(loader_keep_fraction - (1.0 - replace_fraction)) > 0.01:
        raise ValueError(
            "ITA mask direction mismatch: "
            f"replace={replace_fraction:.4f}, loader_keep={loader_keep_fraction:.4f}"
        )
    if abs(loader_replace_fraction - replace_fraction) > 0.01:
        raise ValueError(
            "ITA loss-mask direction mismatch: "
            f"disk_replace={replace_fraction:.4f}, "
            f"loader_replace={loader_replace_fraction:.4f}"
        )
    loader_keep = np.asarray(sample["agn_mask"], dtype=np.float32)[0]
    if float(np.abs(loader_keep + loader_replace - 1.0).max()) > 1e-6:
        raise ValueError("ITA keep and replace masks are not exact complements")
    return {
        "length": len(dataset),
        "keys": sorted(sample),
        "agn_mask_direction": "keep",
        "loss_mask_direction": "replace",
        "replace_fraction": replace_fraction,
        "loader_keep_fraction": loader_keep_fraction,
        "loader_replace_fraction": loader_replace_fraction,
    }


def _probe_idm(repo: Path, layout: Path, split: str) -> dict[str, object]:
    with repo_import(repo):
        module = importlib.import_module("m2h_dataset")
        dataset = module.M2HDataset(
            dataroot_path=str(layout),
            phase=split,
            order="paired",
            size=(512, 512),
        )
        sample = dataset[0]
    if tuple(sample["cloth"].shape) != (1, 3, 224, 224):
        raise ValueError(f"IDM CLIP identity shape is {tuple(sample['cloth'].shape)}")
    if np.array_equal(sample["image"].numpy(), sample["source"].numpy()):
        raise ValueError("IDM target and mannequin source tensors are identical")
    return {
        "length": len(dataset),
        "keys": sorted(sample),
        "identity_clip": _shape(sample["cloth"]),
        "source": _shape(sample["source"]),
    }


def _probe_mcld(repo: Path, layout: Path, split: str, config: Path) -> dict[str, object]:
    with repo_import(repo):
        omega = importlib.import_module("omegaconf").OmegaConf
        module = importlib.import_module("src.dataset.deepfashion_dataset")
        cfg = omega.load(str(config))
        cfg.data.root_dir = str(layout) + "/"
        train_dataset, test_dataset = module.get_deepfashion_dataset(
            cfg,
            double_clip=True,
            use_face_emb=True,
        )
        dataset = train_dataset if split == "train" else test_dataset
        sample = dataset[0]
    face = np.asarray(sample["face_emb"], dtype=np.float32)
    if face.shape != (1, 512) or not np.isfinite(face).all():
        raise ValueError(f"MCLD face embedding shape/value failure: {face.shape}")
    if float(np.linalg.norm(face)) <= 0.0:
        raise ValueError("MCLD face embedding is missing (all zeros)")
    return {
        "length": len(dataset),
        "keys": sorted(sample),
        "face_embedding": list(face.shape),
        "face_embedding_norm": float(np.linalg.norm(face)),
    }


def _probe_omini(
    repo: Path,
    manifest: Path,
    prepared_root: Path,
    config: Path,
) -> dict[str, object]:
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    dataset_cfg = cfg["train"]["dataset"]
    with repo_import(repo):
        module = importlib.import_module("omini.train_flux.train_m2h")
        dataset = module.M2HManifestDataset(
            manifest=str(manifest),
            prepared_root=str(prepared_root),
            target_size=tuple(dataset_cfg["target_size"]),
            condition_size=tuple(dataset_cfg["condition_size"]),
            base_condition=dataset_cfg.get("base_condition", "agnostic"),
            drop_text_prob=0.0,
            drop_image_prob=0.0,
        )
        sample = dataset[0]
    expected = {f"condition_{index}" for index in range(3)}
    if missing := sorted(expected - set(sample)):
        raise KeyError(f"OminiControl sample missing keys: {missing}")
    return {
        "length": len(dataset),
        "keys": sorted(sample),
        "condition_types": [sample[f"condition_type_{i}"] for i in range(3)],
    }


def probe(
    *,
    method: str,
    project_root: Path,
    layout_root: Path,
    prepared_root: Path,
    kind: str,
    profile: str,
) -> dict[str, object]:
    split = "train" if kind == "paired" else "test"
    repos = project_root / "repos"
    layout = layout_root / "low" / kind / method
    config = project_root / "configs" / "methods" / f"{method}_{profile}.yaml"
    if method == "refton":
        details = _probe_refton(repos / "RefTon", layout, split)
    elif method == "ita_mdt":
        details = _probe_ita(repos / "ITA-MDT", layout, split)
    elif method == "idm_vton":
        details = _probe_idm(repos / "IDM-VTON", layout, split)
    elif method == "mcld":
        details = _probe_mcld(repos / "MCLD", layout, split, config)
    elif method == "ominicontrol":
        manifest_split = "train" if kind == "paired" else "counterfactual"
        details = _probe_omini(
            repos / "OminiControl",
            prepared_root / "low" / manifest_split / "samples.jsonl",
            prepared_root,
            config,
        )
    else:
        raise ValueError(f"unknown method: {method}")
    return {"status": "pass", "method": method, "kind": kind, "details": details}


def main() -> None:
    parser = argparse.ArgumentParser(description="Load one M2H sample through an adapted baseline.")
    parser.add_argument("--method", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--layout-root", required=True)
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--kind", choices=("paired", "counterfactual"), default="paired")
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    args = parser.parse_args()
    report = probe(
        method=args.method,
        project_root=Path(args.project_root),
        layout_root=Path(args.layout_root),
        prepared_root=Path(args.prepared_root),
        kind=args.kind,
        profile=args.profile,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
