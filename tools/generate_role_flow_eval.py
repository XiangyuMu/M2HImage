#!/usr/bin/env python3
"""Generate role-flow images from a frozen evaluation manifest.

Model imports are deliberately delayed until after manifest validation so
``--dry-run`` can verify protocol and cache coverage without loading FLUX.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"manifest has no rows: {path}")
    required = {"mid", "jid", "seed", "split", "mannequin_path", "human_path"}
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(f"manifest row {index} missing {sorted(missing)}")
        if row["split"] not in {"val", "test"}:
            raise ValueError(f"manifest row {index} has unsupported split={row['split']!r}")
    return payload


def select_rows(payload: dict[str, Any], split: str, limit: int | None) -> list[dict[str, Any]]:
    rows = [dict(row) for row in payload["rows"] if not split or row["split"] == split]
    rows.sort(key=lambda row: (row["split"], str(row["mid"]), str(row["jid"]), int(row["seed"])))
    return rows[:limit] if limit is not None else rows


def cache_paths(cfg: dict[str, Any]) -> tuple[Path, Path]:
    cache = Path(cfg["data"]["cache_dir"])
    return cache / "samples", cache / "text" / "prompt.npz"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dry_run(cfg: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    sample_dir, text_path = cache_paths(cfg)
    missing_images: list[str] = []
    missing_cache: list[str] = []
    hash_mismatches: list[str] = []
    for row in rows:
        for key in ("mannequin_path", "human_path"):
            image_path = Path(row[key])
            if not image_path.is_file():
                missing_images.append(str(image_path))
            else:
                hash_key = "mannequin_sha256" if key == "mannequin_path" else "human_sha256"
                expected = str(row.get(hash_key, ""))
                if expected and sha256_file(image_path) != expected:
                    hash_mismatches.append(str(image_path))
        for sample_id in (str(row["mid"]), str(row["jid"])):
            path = sample_dir / f"{sample_id}.npz"
            if not path.is_file():
                missing_cache.append(str(path))
    return {
        "rows": len(rows),
        "missing_source_images": sorted(set(missing_images)),
        "missing_cache": sorted(set(missing_cache)),
        "source_hash_mismatches": sorted(set(hash_mismatches)),
        "text_cache": str(text_path),
        "text_cache_exists": text_path.is_file(),
        "status": "pass" if not missing_images and not missing_cache and not hash_mismatches and text_path.is_file() else "failed",
    }


def generate_rows(cfg: dict[str, Any], checkpoint: Path, rows: list[dict[str, Any]], output_dir: Path, device: str, overwrite: bool) -> int:
    import torch
    from conditions import choose_dtype, seed_everything
    from eval_b2 import make_cf_batch
    from eval_watcher import decode_tokens, generate
    from train_paired import WarmupFlowModel, load_checkpoint, load_components

    seed_everything(int(cfg["experiment"]["seed"]))
    torch_device = torch.device(device)
    dtype = choose_dtype(cfg["model"]["precision"])
    transformer, controlnet, vae, adapter, pulid, _ = load_components(cfg, torch_device, dtype)
    if vae is None:
        from diffusers import AutoencoderKL

        vae = AutoencoderKL.from_pretrained(
            cfg["model"]["base"], subfolder="vae", torch_dtype=dtype, local_files_only=True
        ).to(torch_device)
    model = WarmupFlowModel(transformer, controlnet, adapter, pulid, cfg)
    load_checkpoint(checkpoint, model)
    model.eval()
    sample_dir, text_path = cache_paths(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for row in rows:
        mid, jid, seed = str(row["mid"]), str(row["jid"]), int(row["seed"])
        out = output_dir / f"{mid}__id{jid}__seed{seed}.png"
        if out.exists() and not overwrite:
            continue
        batch = make_cf_batch(sample_dir, text_path, mid, jid)
        tokens = generate(model, batch, int(cfg["eval"]["generate_steps"]), seed=seed, device=torch_device, dtype=dtype)
        decode_tokens(vae, tokens, cfg["data"]["resolution"]).save(out)
        written += 1
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", choices=("", "val", "test"), default="")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate protocol and cache coverage without loading FLUX")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    from conditions import load_yaml

    payload = load_manifest(args.manifest)
    rows = select_rows(payload, args.split, args.limit)
    if not rows:
        raise ValueError("no rows selected")
    cfg = load_yaml(args.config)
    if args.dry_run:
        report = validate_dry_run(cfg, rows)
        report.update({"manifest": str(args.manifest.resolve()), "config": str(args.config.resolve()), "split": args.split or "all"})
        print(json.dumps(report, indent=2))
        return 0 if report["status"] == "pass" else 2
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint directory is missing: {args.checkpoint}")
    written = generate_rows(cfg, args.checkpoint, rows, args.output_dir, args.device, args.overwrite)
    print(json.dumps({"status": "ok", "rows": len(rows), "written": written, "output_dir": str(args.output_dir.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
