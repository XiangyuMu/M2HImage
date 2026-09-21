#!/usr/bin/env python3
"""Generate role-flow images from a frozen evaluation manifest.

Model imports are deliberately delayed until after manifest validation so
``--dry-run`` can verify protocol and cache coverage without loading FLUX.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
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


def expected_filename(row: dict[str, Any]) -> str:
    return f"{row['mid']}__id{row['jid']}__seed{int(row['seed'])}.png"


def cache_paths(cfg: dict[str, Any]) -> tuple[Path, Path]:
    cache = Path(cfg["data"]["cache_dir"])
    return cache / "samples", cache / "text" / "prompt.npz"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _signature_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the stable input portion of a generation provenance payload.

    ``finalize_tool`` describes the tool that wrote the provenance file. It is
    useful audit metadata, but changing it must not invalidate generated
    images that were produced from the same inputs.
    """
    signed = {
        key: payload[key]
        for key in ("schema_version", "inputs", "selection", "generation", "code")
        if key in payload
    }
    code = signed.get("code")
    if isinstance(code, dict):
        signed["code"] = {key: value for key, value in code.items() if key != "finalize_tool"}
    return signed


def _legacy_signature_matches(payload: dict[str, Any]) -> bool:
    """Recognize provenance written before ``finalize_tool`` was unsigned."""
    stored = payload.get("input_signature_sha256")
    if not isinstance(stored, str) or not stored:
        return False
    signed = {
        key: payload[key]
        for key in ("schema_version", "inputs", "selection", "generation", "code")
        if key in payload
    }
    return canonical_sha256(signed) == stored


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def current_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit or None


def tool_fingerprint() -> dict[str, Any]:
    script = Path(__file__).resolve()
    return {
        "path": str(script),
        "sha256": sha256_file(script),
        "git_commit": current_git_commit(),
        "source": "current_tool",
    }


def generation_tool_fingerprint(generation_git_commit: str | None, generation_script_sha256: str | None) -> dict[str, Any]:
    current = tool_fingerprint()
    if generation_git_commit or generation_script_sha256:
        return {
            "capture_mode": "retrospective",
            "git_commit": generation_git_commit,
            "script_sha256": generation_script_sha256,
            "source": "operator_supplied",
        }
    return {
        "capture_mode": "live",
        "git_commit": current["git_commit"],
        "script_sha256": current["sha256"],
        "source": "current_tool",
    }


def checkpoint_fingerprint(path: Path) -> dict[str, Any]:
    if path.is_file():
        return {"path": str(path.resolve()), "type": "file", "sha256": sha256_file(path)}
    if not path.is_dir():
        raise FileNotFoundError(f"checkpoint is missing: {path}")
    files: list[dict[str, Any]] = []
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        rel = item.relative_to(path).as_posix()
        files.append({"path": rel, "size": item.stat().st_size, "sha256": sha256_file(item)})
    if not files:
        raise ValueError(f"checkpoint directory has no files: {path}")
    return {
        "path": str(path.resolve()),
        "type": "directory",
        "sha256": canonical_sha256(files),
        "files": files,
    }


def _manifest_content_sha(payload: dict[str, Any]) -> str:
    content = payload.get("content_sha256")
    if not isinstance(content, str) or not content:
        return canonical_sha256({"schema_version": payload.get("schema_version"), "provenance": payload.get("provenance"), "rows": payload["rows"]})
    return content


def _row_key(row: dict[str, Any]) -> dict[str, Any]:
    return {"split": row["split"], "mid": str(row["mid"]), "jid": str(row["jid"]), "seed": int(row["seed"])}


def build_generation_inputs(
    *,
    cfg: dict[str, Any],
    config: Path,
    checkpoint: Path,
    manifest: Path,
    manifest_payload: dict[str, Any],
    rows: list[dict[str, Any]],
    split: str,
    limit: int | None,
    generation_git_commit: str | None = None,
    generation_script_sha256: str | None = None,
) -> dict[str, Any]:
    expected = [expected_filename(row) for row in rows]
    if len(expected) != len(set(expected)):
        raise ValueError("selected rows produce duplicate output filenames")
    generate_steps = int(cfg["eval"]["generate_steps"])
    resolution = cfg["data"]["resolution"]
    body = {
        "schema_version": "m2h-role-flow-generation-v2",
        "inputs": {
            "config": {"path": str(config.resolve()), "sha256": sha256_file(config)},
            "checkpoint": checkpoint_fingerprint(checkpoint),
            "manifest": {
                "path": str(manifest.resolve()),
                "sha256": sha256_file(manifest),
                "content_sha256": _manifest_content_sha(manifest_payload),
            },
        },
        "selection": {
            "split": split or "all",
            "limit": limit,
            "selected_row_count": len(rows),
            "row_keys": [_row_key(row) for row in rows],
            "expected_filenames": expected,
        },
        "generation": {
            "generate_steps": generate_steps,
            "resolution": resolution,
        },
        "code": {
            "generation": generation_tool_fingerprint(generation_git_commit, generation_script_sha256),
        },
    }
    body["input_signature_sha256"] = canonical_sha256(body)
    body["code"]["finalize_tool"] = tool_fingerprint()
    return body


def read_png(path: Path) -> dict[str, Any]:
    from PIL import Image

    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
    except Exception as exc:
        return {
            "filename": path.name,
            "path": str(path.resolve()),
            "readable": False,
            "status": "unreadable",
            "error": str(exc),
        }
    return {
        "filename": path.name,
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "width": width,
        "height": height,
        "mode": mode,
        "readable": True,
        "status": "ok",
    }


def _expected_size(inputs: dict[str, Any]) -> tuple[int, int]:
    resolution = inputs["generation"]["resolution"]
    if isinstance(resolution, int):
        return resolution, resolution
    if isinstance(resolution, (list, tuple)) and len(resolution) == 2:
        return int(resolution[0]), int(resolution[1])
    # C2 config records the frozen resolution as a width/height mapping.
    if isinstance(resolution, dict) and {"width", "height"} <= set(resolution):
        return int(resolution["width"]), int(resolution["height"])
    raise ValueError(f"unsupported generation resolution: {resolution!r}")


def inspect_output_dir(
    output_dir: Path,
    expected: list[str],
    *,
    require_complete: bool,
    row_keys: list[dict[str, Any]] | None = None,
    expected_size: tuple[int, int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    expected_set = set(expected)
    metadata = {name: {"row_key": row_key} for name, row_key in zip(expected, row_keys or [], strict=False)}
    pngs = sorted(path.name for path in output_dir.glob("*.png")) if output_dir.exists() else []
    failures: list[dict[str, Any]] = []
    extra = [name for name in pngs if name not in expected_set]
    missing = [name for name in expected if name not in pngs]
    for name in extra:
        failures.append({"filename": name, "status": "extra_png"})
    if require_complete:
        for name in missing:
            failures.append({"filename": name, "status": "missing_png"})
    images: list[dict[str, Any]] = []
    for name in expected:
        path = output_dir / name
        if path.exists():
            record = read_png(path)
            record.update(metadata.get(name, {}))
            images.append(record)
            if not record.get("readable"):
                failures.append({"filename": name, "status": "unreadable", "error": record.get("error", "")})
                continue
            if record.get("mode") != "RGB":
                failures.append({"filename": name, "status": "non_rgb", "mode": record.get("mode")})
            if expected_size and (record.get("width"), record.get("height")) != expected_size:
                failures.append({
                    "filename": name,
                    "status": "size_mismatch",
                    "expected_width": expected_size[0],
                    "expected_height": expected_size[1],
                    "actual_width": record.get("width"),
                    "actual_height": record.get("height"),
                })
    return images, failures


def load_generation_provenance(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"generation provenance is not an object: {path}")
    return payload


def write_generation_provenance(
    output_dir: Path,
    inputs: dict[str, Any],
    *,
    status: str,
    images: list[dict[str, Any]] | None = None,
    failures: list[dict[str, Any]] | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    now = utc_now()
    existing = load_generation_provenance(output_dir / "generation_provenance.json")
    payload = dict(inputs)
    payload.update({
        "status": status,
        "created_at": existing.get("created_at", now) if existing else now,
        "updated_at": now,
        "device": device,
        "images": images or [],
        "failures": failures or [],
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "generation_provenance.json"
    temporary = output_dir / f".{destination.name}.tmp"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return payload


def validate_or_prepare_output(output_dir: Path, inputs: dict[str, Any], *, finalize_only: bool, device: str) -> dict[str, Any]:
    expected = list(inputs["selection"]["expected_filenames"])
    expected_size = _expected_size(inputs)
    prov_path = output_dir / "generation_provenance.json"
    existing = load_generation_provenance(prov_path)
    has_pngs = output_dir.exists() and any(output_dir.glob("*.png"))
    if existing and existing.get("input_signature_sha256") != inputs["input_signature_sha256"]:
        compatible = (
            canonical_sha256(_signature_payload(existing)) == inputs["input_signature_sha256"]
            and _legacy_signature_matches(existing)
        )
        if not compatible:
            raise ValueError(f"existing generation_provenance.json does not match current inputs: {prov_path}")
    if has_pngs and not existing and not finalize_only:
        raise ValueError(f"output dir already contains PNGs but no matching generation_provenance.json: {output_dir}")
    images, failures = inspect_output_dir(
        output_dir,
        expected,
        require_complete=finalize_only,
        row_keys=list(inputs["selection"]["row_keys"]),
        expected_size=expected_size,
    )
    if failures:
        status = "failed"
        if existing or finalize_only:
            write_generation_provenance(output_dir, inputs, status=status, images=images, failures=failures, device=device)
        raise ValueError(f"output validation failed for {output_dir}: {failures[:5]}")
    if finalize_only:
        return write_generation_provenance(output_dir, inputs, status="complete", images=images, failures=[], device=device)
    return write_generation_provenance(output_dir, inputs, status="in_progress", images=images, failures=[], device=device)


def finalize_output(output_dir: Path, inputs: dict[str, Any], *, device: str) -> dict[str, Any]:
    images, failures = inspect_output_dir(
        output_dir,
        list(inputs["selection"]["expected_filenames"]),
        require_complete=True,
        row_keys=list(inputs["selection"]["row_keys"]),
        expected_size=_expected_size(inputs),
    )
    if len(images) != int(inputs["selection"]["selected_row_count"]):
        failures.append({
            "status": "row_count_mismatch",
            "expected": int(inputs["selection"]["selected_row_count"]),
            "actual": len(images),
        })
    status = "complete" if not failures else "failed"
    payload = write_generation_provenance(output_dir, inputs, status=status, images=images, failures=failures, device=device)
    if failures:
        raise ValueError(f"final output validation failed for {output_dir}: {failures[:5]}")
    return payload


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
    parser.add_argument("--finalize-only", action="store_true", help="Validate existing PNGs and write generation_provenance.json without loading FLUX")
    parser.add_argument("--generation-git-commit", default=None, help="Actual generation commit for retrospective finalize-only imports")
    parser.add_argument("--generation-script-sha256", default=None, help="Actual generation script SHA-256 for retrospective finalize-only imports")
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
    inputs = build_generation_inputs(
        cfg=cfg,
        config=args.config,
        checkpoint=args.checkpoint,
        manifest=args.manifest,
        manifest_payload=payload,
        rows=rows,
        split=args.split,
        limit=args.limit,
        generation_git_commit=args.generation_git_commit,
        generation_script_sha256=args.generation_script_sha256,
    )
    validate_or_prepare_output(args.output_dir, inputs, finalize_only=args.finalize_only, device=args.device)
    if args.finalize_only:
        print(json.dumps({
            "status": "ok",
            "mode": "finalize-only",
            "rows": len(rows),
            "output_dir": str(args.output_dir.resolve()),
            "provenance": str((args.output_dir / "generation_provenance.json").resolve()),
        }, indent=2))
        return 0
    written = generate_rows(cfg, args.checkpoint, rows, args.output_dir, args.device, args.overwrite)
    final = finalize_output(args.output_dir, inputs, device=args.device)
    print(json.dumps({
        "status": "ok",
        "rows": len(rows),
        "written": written,
        "output_dir": str(args.output_dir.resolve()),
        "provenance": str((args.output_dir / "generation_provenance.json").resolve()),
        "input_signature_sha256": final["input_signature_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
