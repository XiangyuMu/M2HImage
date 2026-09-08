#!/usr/bin/env python3
"""B-D0 input-side probe for M RGB -> FLUX VAE roundtrip diagnostics.

This script is intentionally not a formal B result. It reads only mannequin
images from a fixed old-val split (or explicit ids JSON), creates FASHN garment
masks from M RGB, roundtrips M through the FLUX VAE with deterministic posterior
mode, and reports image/region reconstruction metrics.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import math
import os
import re
import resource
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


DEFAULT_REPO_ROOT = Path("/data/muxiangyu/pythonPrograms/M2HImage")
DEFAULT_DATA_ROOT = Path("/data/muxiangyu/datasets/M2HImage/M2H_Final_v2")
DEFAULT_VAE_DIR = DEFAULT_REPO_ROOT / "models/hf/black-forest-labs/FLUX.1-dev/vae"
DEFAULT_FASHN_DIR = DEFAULT_REPO_ROOT / "models/hf/fashn-ai/fashn-human-parser"
DEFAULT_OUT_ROOT = DEFAULT_DATA_ROOT / "experiments/m2h_ab_objective_v1/b_d0_probe"
DEFAULT_GARMENT_LABELS = (3, 4, 5, 6, 7, 10)
CANONICAL_SPLIT = "old-val"
CANONICAL_SPLIT_FILE = "splits/val.txt"
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class ProbePaths:
    run_dir: Path
    metrics_jsonl: Path
    summary_json: Path
    masks_dir: Path
    recon_dir: Path


def read_ids_from_split(data_root: Path, split_file: str = CANONICAL_SPLIT_FILE) -> list[str]:
    path = data_root / split_file
    ids = [validate_sample_id(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return ids


def validate_sample_id(value: str) -> str:
    sample_id = str(value).strip()
    if not sample_id:
        raise ValueError("empty sample id")
    if sample_id in {".", ".."} or "/" in sample_id or "\\" in sample_id:
        raise ValueError(f"unsafe sample id blocks path traversal: {sample_id!r}")
    if Path(sample_id).name != sample_id or not SAFE_ID_RE.fullmatch(sample_id):
        raise ValueError(f"unsafe sample id blocks path traversal: {sample_id!r}")
    return sample_id


def read_explicit_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, dict):
        for key in ("ids", "mannequin_ids", "mids"):
            if key in payload:
                values = payload[key]
                break
        else:
            raise ValueError(f"{path} must contain a list or one of ids/mannequin_ids/mids")
    else:
        raise ValueError(f"{path} must contain a JSON list or object")
    ids = [validate_sample_id(str(value)) for value in values if str(value).strip()]
    if not ids:
        raise ValueError(f"{path} contains no usable ids")
    return ids


def assert_subset_of_old_val(explicit_ids: Iterable[str], old_val_ids: Iterable[str]) -> None:
    old_val = set(old_val_ids)
    missing = [sample_id for sample_id in explicit_ids if sample_id not in old_val]
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"--ids-json contains ids outside old-val split: {preview}")


def select_probe_ids(ids: Iterable[str], limit: int, seed: int, preserve_order: bool = False) -> list[str]:
    unique_ids = list(dict.fromkeys(str(value).strip() for value in ids if str(value).strip()))
    if limit <= 0:
        raise ValueError("--limit must be positive")
    if len(unique_ids) <= limit:
        return unique_ids
    if preserve_order:
        return unique_ids[:limit]
    rng = np.random.default_rng(np.uint64(seed))
    positions = np.sort(rng.choice(len(unique_ids), size=limit, replace=False))
    return [unique_ids[int(position)] for position in positions]


def find_existing_image(root: Path, stem: str) -> Path:
    for suffix in (".png", ".jpg", ".jpeg", ".webp"):
        path = root / f"{stem}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"missing image for id={stem} under {root}")


def make_unique_run_dir(out_root: Path, run_name: str | None = None) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    base = run_name or dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    for attempt in range(1000):
        name = base if attempt == 0 else f"{base}_{attempt:03d}"
        path = out_root / name
        try:
            path.mkdir(parents=False, exist_ok=False)
            return path
        except FileExistsError:
            continue
    raise FileExistsError(f"could not allocate a unique run directory under {out_root}")


def prepare_probe_paths(out_root: Path, run_name: str | None = None) -> ProbePaths:
    run_dir = make_unique_run_dir(out_root, run_name)
    masks_dir = run_dir / "masks"
    recon_dir = run_dir / "recon_samples"
    masks_dir.mkdir()
    recon_dir.mkdir()
    return ProbePaths(
        run_dir=run_dir,
        metrics_jsonl=run_dir / "per_image_metrics.jsonl",
        summary_json=run_dir / "summary.json",
        masks_dir=masks_dir,
        recon_dir=recon_dir,
    )


def image_to_uint8_array(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def binary_mask_from_labels(label_map: np.ndarray, labels: Iterable[int]) -> np.ndarray:
    return np.isin(np.asarray(label_map), np.asarray(tuple(int(x) for x in labels))).astype(np.uint8)


def masked_mse(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    af = np.asarray(a, dtype=np.float32) / 255.0
    bf = np.asarray(b, dtype=np.float32) / 255.0
    diff2 = (af - bf) ** 2
    if mask is None:
        return float(np.mean(diff2))
    weights = np.asarray(mask, dtype=bool)
    if weights.shape != diff2.shape[:2] or not np.any(weights):
        return float("nan")
    return float(np.mean(diff2[weights]))


def to_json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(item) for item in value]
    return value


def psnr_from_mse(mse: float) -> float:
    if math.isnan(mse):
        return float("nan")
    if mse <= 0.0:
        return float("inf")
    return float(10.0 * math.log10(1.0 / mse))


def _gaussian_blur_rgb(image: np.ndarray, sigma: float) -> np.ndarray:
    try:
        import cv2

        return cv2.GaussianBlur(np.asarray(image, dtype=np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
    except Exception:
        from PIL import ImageFilter

        blurred = Image.fromarray(np.asarray(image, dtype=np.uint8)).filter(ImageFilter.GaussianBlur(radius=float(sigma)))
        return np.asarray(blurred, dtype=np.float32)


def high_frequency_mse(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None, sigma: float = 1.5) -> float:
    af = np.asarray(a, dtype=np.float32) / 255.0
    bf = np.asarray(b, dtype=np.float32) / 255.0
    ah = af - (_gaussian_blur_rgb(a, sigma) / 255.0)
    bh = bf - (_gaussian_blur_rgb(b, sigma) / 255.0)
    diff2 = (ah - bh) ** 2
    if mask is None:
        return float(np.mean(diff2))
    weights = np.asarray(mask, dtype=bool)
    if weights.shape != diff2.shape[:2] or not np.any(weights):
        return float("nan")
    return float(np.mean(diff2[weights]))


def mask_bbox(mask: np.ndarray, pad: int = 8) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if len(xs) == 0:
        return None
    h, w = mask.shape[:2]
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(w, int(xs.max()) + pad + 1)
    y1 = min(h, int(ys.max()) + pad + 1)
    return x0, y0, x1, y1


def summarize_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_keys = [
        "mse_full",
        "psnr_full",
        "hf_mse_full",
        "mse_garment",
        "psnr_garment",
        "hf_mse_garment",
        "lpips_garment_crop",
    ]
    summary: dict[str, Any] = {
        "count_total": len(rows),
        "count_ok": sum(1 for row in rows if row.get("status") == "ok"),
        "count_failed": sum(1 for row in rows if row.get("status") != "ok"),
        "count_parse_failed": sum(1 for row in rows if row.get("status") == "parse_failed"),
        "failure_status_counts": {},
    }
    for row in rows:
        status = str(row.get("status", "unknown"))
        if status != "ok":
            summary["failure_status_counts"][status] = int(summary["failure_status_counts"].get(status, 0)) + 1
    for key in metric_keys:
        values = np.asarray(
            [float(row[key]) for row in rows if key in row and np.isfinite(float(row[key]))],
            dtype=np.float64,
        )
        if len(values):
            summary[key] = {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "std": float(values.std(ddof=0)),
                "n": int(len(values)),
            }
    return summary


def add_repo_to_path(repo_root: Path) -> None:
    path = str(repo_root)
    if path not in sys.path:
        sys.path.insert(0, path)


def load_parser(repo_root: Path, model_dir: Path, device: str, input_width: int, input_height: int) -> Any:
    add_repo_to_path(repo_root)
    from metrics_v2.parsing import FashnParser

    return FashnParser(model_dir, device=device, input_size=(input_width, input_height))


def load_vae(vae_dir: Path, device: str, dtype_name: str) -> Any:
    import torch
    from diffusers import AutoencoderKL

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype_name]
    vae = AutoencoderKL.from_pretrained(str(vae_dir), torch_dtype=dtype, local_files_only=True)
    vae.eval().requires_grad_(False).to(torch.device(device))
    return vae


def pil_to_vae_tensor(image: Image.Image, device: Any, dtype: Any) -> Any:
    import torch

    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return (tensor * 2.0 - 1.0).to(device=device, dtype=dtype)


def vae_tensor_to_uint8(tensor: Any) -> np.ndarray:
    image = ((tensor.detach().float().cpu().clamp(-1, 1)[0] + 1.0) * 127.5).round()
    return image.byte().permute(1, 2, 0).numpy()


def flux_scale_latents(latents: Any, scaling_factor: float, shift_factor: float) -> Any:
    return (latents - shift_factor) * scaling_factor


def flux_unscale_latents(scaled_latents: Any, scaling_factor: float, shift_factor: float) -> Any:
    return (scaled_latents / scaling_factor) + shift_factor


def encode_decode_vae_roundtrip(vae: Any, image: Image.Image, device: str, dtype_name: str) -> np.ndarray:
    import torch

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype_name]
    tensor = pil_to_vae_tensor(image, torch.device(device), dtype)
    with torch.inference_mode():
        posterior = vae.encode(tensor).latent_dist
        latent = posterior.mode()
        scaled = flux_scale_latents(latent, vae.config.scaling_factor, vae.config.shift_factor)
        decoded = vae.decode(
            flux_unscale_latents(scaled, vae.config.scaling_factor, vae.config.shift_factor),
            return_dict=False,
        )[0]
    return vae_tensor_to_uint8(decoded)


def lpips_version() -> str | None:
    try:
        return importlib.metadata.version("lpips")
    except importlib.metadata.PackageNotFoundError:
        return None


def load_lpips(device: str) -> Any | None:
    try:
        import lpips
        import torch

        model = lpips.LPIPS(net="alex")
        model.eval().requires_grad_(False).to(torch.device(device))
        return model
    except Exception:
        return None


def lpips_on_bbox(lpips_model: Any, a: np.ndarray, b: np.ndarray, bbox: tuple[int, int, int, int], device: str) -> float:
    import torch

    x0, y0, x1, y1 = bbox
    crops = []
    for image in (a[y0:y1, x0:x1], b[y0:y1, x0:x1]):
        arr = np.asarray(image, dtype=np.float32) / 255.0
        crops.append(torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0)
    with torch.inference_mode():
        value = lpips_model(crops[0].to(device), crops[1].to(device))
    return float(value.detach().float().cpu().reshape(-1)[0])


def save_side_by_side(path: Path, original: np.ndarray, recon: np.ndarray, mask: np.ndarray) -> None:
    mask_rgb = np.repeat((np.asarray(mask, dtype=np.uint8) * 255)[:, :, None], 3, axis=2)
    panel = np.concatenate([original, recon, mask_rgb], axis=1)
    Image.fromarray(panel).save(path)


def write_jsonl_row(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(to_json_safe(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    handle.flush()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def optional_sha256(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    return sha256_file(path)


def model_hashes(vae_dir: Path, fashn_model_dir: Path) -> dict[str, Any]:
    candidates = {
        "flux_vae_config": vae_dir / "config.json",
        "flux_vae_weights": vae_dir / "diffusion_pytorch_model.safetensors",
        "fashn_config": fashn_model_dir / "config.json",
        "fashn_weights": fashn_model_dir / "model.safetensors",
    }
    return {name: {"path": str(path), "sha256": optional_sha256(path)} for name, path in candidates.items()}


def build_summary(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    selected_ids: list[str],
    paths: ProbePaths,
    lpips_ver: str | None,
    lpips_enabled: bool,
    hashes: dict[str, Any],
    gpu_stats: dict[str, Any],
    started_at: float,
) -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "experiment": "B-D0 input-side VAE probe",
        "formal_result": False,
        "warning": "Diagnostic only: no H, no generated base Y0, no training, not a formal B result.",
        "source_boundary": {
            "source": "explicit_ids_json" if args.ids_json else CANONICAL_SPLIT,
            "split_file": None if args.ids_json else CANONICAL_SPLIT_FILE,
            "forbidden_sources": ["old-test", "final", "images/human", "generated base outputs"],
            "mask_source": "FASHN parser on M RGB only",
        },
        "paths": {
            "run_dir": str(paths.run_dir),
            "metrics_jsonl": str(paths.metrics_jsonl),
            "summary_json": str(paths.summary_json),
            "masks_dir": str(paths.masks_dir),
            "recon_dir": str(paths.recon_dir),
        },
        "config": {
            "repo_root": str(args.repo_root),
            "data_root": str(args.data_root),
            "vae_dir": str(args.vae_dir),
            "fashn_model_dir": str(args.fashn_model_dir),
            "device": args.device,
            "dtype": args.dtype,
            "limit": args.limit,
            "seed": args.seed,
            "parser_batch_size": args.parser_batch_size,
            "parser_input_width": args.parser_input_width,
            "parser_input_height": args.parser_input_height,
            "save_recon_limit": args.save_recon_limit,
            "lpips_mode": args.lpips_mode,
        },
        "selected_ids": selected_ids,
        "lpips": {"available": lpips_ver is not None, "enabled": lpips_enabled, "version": lpips_ver},
        "hashes": hashes,
        "metrics": summarize_metrics(rows),
        "resources": {
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "elapsed_seconds": round(time.time() - started_at, 3),
            "max_rss_kb": int(usage.ru_maxrss),
            "cuda": gpu_stats,
        },
    }


def init_cuda_meter(device: str) -> dict[str, Any]:
    try:
        import torch

        torch_device = torch.device(device)
        if torch_device.type != "cuda" or not torch.cuda.is_available():
            return {"enabled": False, "device": device}
        torch.cuda.set_device(torch_device)
        torch.cuda.reset_peak_memory_stats(torch_device)
        torch.cuda.synchronize(torch_device)
        return {
            "enabled": True,
            "device": str(torch_device),
            "gpu_seconds_sum": 0.0,
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
        }
    except Exception as exc:
        return {"enabled": False, "device": device, "error": repr(exc)}


def start_cuda_event(meter: dict[str, Any]) -> Any | None:
    if not meter.get("enabled"):
        return None
    import torch

    torch_device = torch.device(str(meter["device"]))
    torch.cuda.synchronize(torch_device)
    start = torch.cuda.Event(enable_timing=True)
    start.record()
    return start


def stop_cuda_event(meter: dict[str, Any], start: Any | None) -> float | None:
    if not meter.get("enabled") or start is None:
        return None
    import torch

    torch_device = torch.device(str(meter["device"]))
    end = torch.cuda.Event(enable_timing=True)
    end.record()
    torch.cuda.synchronize(torch_device)
    elapsed = float(start.elapsed_time(end) / 1000.0)
    meter["gpu_seconds_sum"] = float(meter.get("gpu_seconds_sum", 0.0)) + elapsed
    meter["peak_allocated_bytes"] = max(int(meter.get("peak_allocated_bytes", 0)), int(torch.cuda.max_memory_allocated(torch_device)))
    meter["peak_reserved_bytes"] = max(int(meter.get("peak_reserved_bytes", 0)), int(torch.cuda.max_memory_reserved(torch_device)))
    return elapsed


def run_probe(args: argparse.Namespace) -> tuple[Path, int]:
    started_at = time.time()
    paths = prepare_probe_paths(args.out_root, args.run_name)
    old_val_ids = read_ids_from_split(args.data_root)
    ids_source = args.ids_json if args.ids_json else args.data_root / CANONICAL_SPLIT_FILE
    if args.ids_json:
        ids = read_explicit_ids(args.ids_json)
        assert_subset_of_old_val(ids, old_val_ids)
    else:
        ids = old_val_ids
    selected_ids = select_probe_ids(ids, args.limit, args.seed, preserve_order=bool(args.ids_json))

    parser = load_parser(args.repo_root, args.fashn_model_dir, args.device, args.parser_input_width, args.parser_input_height)
    vae = load_vae(args.vae_dir, args.device, args.dtype)
    gpu_stats = init_cuda_meter(args.device)
    lpips_ver = lpips_version()
    lpips_model = None
    if args.lpips_mode != "off" and lpips_ver:
        lpips_model = load_lpips(args.device)
    if args.lpips_mode == "required" and lpips_model is None:
        raise RuntimeError("LPIPS was required but could not be loaded")

    rows: list[dict[str, Any]] = []
    image_root = args.data_root / "images/mannequin"
    with paths.metrics_jsonl.open("w", encoding="utf-8") as handle:
        for index, mid in enumerate(selected_ids):
            row: dict[str, Any] = {
                "index": index,
                "mid": mid,
                "status": "ok",
                "source_split": "explicit_ids_json" if args.ids_json else CANONICAL_SPLIT,
                "mask_source": "fashn_m_rgb",
            }
            gpu_start = start_cuda_event(gpu_stats)
            try:
                image_path = find_existing_image(image_root, mid)
                original = image_to_uint8_array(image_path)
                labels = parser.predict([original])[0]
                mask = binary_mask_from_labels(labels, args.garment_labels)
                Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(paths.masks_dir / f"{mid}.png")
                garment_area = float(mask.mean())
                if garment_area < args.min_garment_area_fraction:
                    raise RuntimeError(
                        f"parse_failed: garment_area_fraction={garment_area:.8f} "
                        f"< min_garment_area_fraction={args.min_garment_area_fraction:.8f}"
                    )
                recon = encode_decode_vae_roundtrip(vae, Image.fromarray(original), args.device, args.dtype)
                if recon.shape != original.shape:
                    raise RuntimeError(f"shape_mismatch: original={original.shape} recon={recon.shape}")

                mse_full = masked_mse(original, recon)
                mse_garment = masked_mse(original, recon, mask)
                row.update(
                    {
                        "image_path": str(image_path),
                        "garment_area_fraction": garment_area,
                        "mse_full": mse_full,
                        "psnr_full": psnr_from_mse(mse_full),
                        "hf_mse_full": high_frequency_mse(original, recon),
                        "mse_garment": mse_garment,
                        "psnr_garment": psnr_from_mse(mse_garment),
                        "hf_mse_garment": high_frequency_mse(original, recon, mask),
                    }
                )
                bbox = mask_bbox(mask)
                if lpips_model is not None and bbox is not None:
                    row["lpips_garment_crop"] = lpips_on_bbox(lpips_model, original, recon, bbox, args.device)
                if index < args.save_recon_limit:
                    save_side_by_side(paths.recon_dir / f"{mid}_original_recon_mask.png", original, recon, mask)
            except Exception as exc:
                status = "parse_failed" if "parse_failed:" in str(exc) else "failed"
                if "shape_mismatch:" in str(exc):
                    status = "shape_mismatch"
                row.update({"status": status, "error": repr(exc)})
            finally:
                gpu_seconds = stop_cuda_event(gpu_stats, gpu_start)
                if gpu_seconds is not None:
                    row["gpu_seconds"] = gpu_seconds
            rows.append(row)
            write_jsonl_row(handle, row)
            print(f"progress {index + 1}/{len(selected_ids)} mid={mid} status={row['status']}", flush=True)

    hashes = {
        "script": {"path": str(Path(__file__).resolve()), "sha256": optional_sha256(Path(__file__).resolve())},
        "ids_source": {"path": str(ids_source), "sha256": optional_sha256(ids_source)},
        "models": model_hashes(args.vae_dir, args.fashn_model_dir),
    }
    summary = build_summary(args, rows, selected_ids, paths, lpips_ver, lpips_model is not None, hashes, gpu_stats, started_at)
    paths.summary_json.write_text(
        json.dumps(to_json_safe(summary), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    failed = int(summary["metrics"]["count_failed"])
    if failed == 0:
        (paths.run_dir / "READY").write_text("ok\n", encoding="utf-8")
    return paths.run_dir, failed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B-D0 M RGB -> FASHN mask + FLUX VAE roundtrip diagnostic")
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--vae-dir", type=Path, default=DEFAULT_VAE_DIR)
    parser.add_argument("--fashn-model-dir", type=Path, default=DEFAULT_FASHN_DIR)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--ids-json", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float32")
    parser.add_argument("--parser-batch-size", type=int, default=1, help="Reserved for manifesting; parser API is called one image at a time for failure accounting.")
    parser.add_argument("--parser-input-width", type=int, default=384)
    parser.add_argument("--parser-input-height", type=int, default=576)
    parser.add_argument("--garment-labels", type=int, nargs="+", default=list(DEFAULT_GARMENT_LABELS))
    parser.add_argument("--min-garment-area-fraction", type=float, default=0.005)
    parser.add_argument("--save-recon-limit", type=int, default=8)
    parser.add_argument(
        "--lpips-mode",
        choices=("off", "auto", "required"),
        default="auto",
        help="off avoids importing LPIPS; auto uses it if installed/loadable; required fails if unavailable.",
    )
    parser.add_argument("--no-lpips", action="store_true", help="Alias for --lpips-mode off.")
    args = parser.parse_args(argv)
    args.repo_root = args.repo_root.resolve()
    args.data_root = args.data_root.resolve()
    args.vae_dir = args.vae_dir.resolve()
    args.fashn_model_dir = args.fashn_model_dir.resolve()
    args.out_root = args.out_root.resolve()
    args.garment_labels = tuple(args.garment_labels)
    if args.ids_json is not None:
        args.ids_json = args.ids_json.resolve()
    if args.no_lpips:
        args.lpips_mode = "off"
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir, failed = run_probe(args)
    print(f"B-D0 diagnostic complete: {run_dir}; failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
