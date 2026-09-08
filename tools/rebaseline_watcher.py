from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image, ImageDraw

from conditions import choose_dtype, find_one, load_yaml, seed_everything
from dataset import PairedWarmupDataset
from eval_watcher import (
    cosine,
    decode_tokens,
    embedding_for_image,
    generate,
    swap_identity,
)
from metrics_v2.parsing import DEFAULT_GARMENT_LABELS, parsing_path
from probe_response_track import compute_response_snapshot, run_watcher_metrics
from train_paired import WarmupFlowModel, load_checkpoint, load_components
from watcher_protocol import watcher_eval_set_from_config


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def same_type_donors(protocol: dict[str, Any]) -> dict[str, str]:
    groups: dict[str, list[str]] = defaultdict(list)
    for row in protocol["samples"]:
        groups[str(row["garment_type"])].append(str(row["id"]))
    return {
        sample_id: ids[(index + 1) % len(ids)]
        for ids in groups.values()
        for index, sample_id in enumerate(ids)
    }


def metric_rows(protocol: dict[str, Any], image_dir: Path) -> list[dict[str, Any]]:
    by_id = {str(row["id"]): row for row in protocol["samples"]}
    return [
        {
            "mid": sample_id,
            "jid": sample_id,
            "seed": 1000 + index,
            "garment_type": str(by_id[sample_id]["garment_type"]),
            "path": image_dir / f"{sample_id}_generated.png",
        }
        for index, sample_id in enumerate(protocol["sample_ids"])
    ]


def evaluate_checkpoint(
    cfg: dict[str, Any],
    checkpoint: Path,
    output_root: Path,
    device_name: str,
    overwrite_images: bool = False,
) -> dict[str, Any]:
    protocol = watcher_eval_set_from_config(cfg)
    seed_everything(int(cfg["experiment"]["seed"]))
    device = torch.device(device_name)
    if device.type != "cuda":
        raise RuntimeError("watcher rebaseline requires CUDA")
    torch.cuda.set_device(device)
    dtype = choose_dtype(cfg["model"]["precision"])
    dataset = PairedWarmupDataset(cfg, "val", require_coverage=True)
    missing = sorted(set(protocol["sample_ids"]) - set(dataset.ids))
    if missing:
        raise RuntimeError(f"frozen watcher ids missing from val dataset: {missing}")

    transformer, controlnet, vae, adapter, pulid, _ = load_components(
        cfg, device, dtype
    )
    if vae is None:
        from diffusers import AutoencoderKL

        vae = AutoencoderKL.from_pretrained(
            cfg["model"]["base"],
            subfolder="vae",
            torch_dtype=dtype,
            local_files_only=True,
        ).to(device)
    model = WarmupFlowModel(transformer, controlnet, adapter, pulid, cfg)
    step = load_checkpoint(checkpoint, model)
    model.eval()
    step_dir = output_root / f"step{step}"
    image_dir = step_dir / "generated"
    image_dir.mkdir(parents=True, exist_ok=True)
    donors = same_type_donors(protocol)
    failures: list[dict[str, str]] = []
    generated_paths: list[Path] = []
    swap_pairs: list[dict[str, Any]] = []

    for index, sample_id in enumerate(protocol["sample_ids"]):
        batch = dataset[dataset.ids.index(sample_id)]
        seed = 1000 + index
        paired_path = image_dir / f"{sample_id}_generated.png"
        donor_id = donors[sample_id]
        swap_path = image_dir / f"{sample_id}_swap_{donor_id}.png"
        try:
            if overwrite_images or not paired_path.exists():
                tokens = generate(
                    model, batch, int(cfg["eval"]["generate_steps"]),
                    seed, device, dtype,
                )
                decode_tokens(vae, tokens, cfg["data"]["resolution"]).save(paired_path)
            generated_paths.append(paired_path)
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {"sample_id": sample_id, "branch": "paired", "error": str(exc)}
            )
            continue
        try:
            donor = dataset[dataset.ids.index(donor_id)]
            if overwrite_images or not swap_path.exists():
                tokens = generate(
                    model, swap_identity(batch, donor),
                    int(cfg["eval"]["generate_steps"]), seed, device, dtype,
                )
                decode_tokens(vae, tokens, cfg["data"]["resolution"]).save(swap_path)
            generated_paths.append(swap_path)
            swap_pairs.append(
                {
                    "sample_id": sample_id,
                    "swap_id": donor_id,
                    "paired_path": paired_path,
                    "swap_path": swap_path,
                }
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {"sample_id": sample_id, "branch": "swap", "error": str(exc)}
            )

    response = compute_response_snapshot(
        cfg, model, dataset, checkpoint, step, device, dtype
    )
    gates = model.adapter.gate_values()
    del model, transformer, controlnet, pulid, adapter, vae
    gc.collect()
    torch.cuda.empty_cache()

    rows = metric_rows(protocol, image_dir)
    missing_paired = [
        str(row["mid"]) for row in rows if not Path(row["path"]).exists()
    ]
    if missing_paired:
        raise RuntimeError(
            f"paired rebaseline generation failed for {missing_paired}"
        )
    metrics = run_watcher_metrics(
        cfg, rows, step_dir / "metrics_v2", device_name
    )
    gc.collect()
    torch.cuda.empty_cache()

    device_index = device.index or 0
    embeddings = {
        str(path): embedding_for_image(path, cfg, device_index)
        for path in generated_paths
    }
    detected = sum(value is not None for value in embeddings.values())
    swap_rows = []
    for row in swap_pairs:
        paired = embeddings.get(str(row["paired_path"]))
        swapped = embeddings.get(str(row["swap_path"]))
        swap_rows.append(
            {
                "sample_id": row["sample_id"],
                "swap_id": row["swap_id"],
                "cosine": (
                    cosine(paired, swapped)
                    if paired is not None and swapped is not None
                    else None
                ),
            }
        )
    valid_cos = [
        float(row["cosine"])
        for row in swap_rows
        if row["cosine"] is not None
    ]
    guards = {
        "face_detection_rate": detected / max(len(generated_paths), 1),
        "face_detection_count": detected,
        "face_image_count": len(generated_paths),
        "swap_cosine_mean": float(np.mean(valid_cos)) if valid_cos else None,
        "swap_cosine_median": float(np.median(valid_cos)) if valid_cos else None,
        "swap_cosine_max": float(np.max(valid_cos)) if valid_cos else None,
        "swap_rows": swap_rows,
    }
    payload = {
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(step),
        "protocol": {
            "version": protocol["protocol_version"],
            "hash": protocol["protocol_hash"],
            "path": protocol["path"],
            "sample_ids": protocol["sample_ids"],
        },
        "gates": gates,
        "response": response["response"],
        "response_by_tau": response["response_by_tau"],
        "response_protocol": response["protocol"],
        "watcher_metrics": metrics,
        "guards": guards,
        "generation_failures": failures,
        "output_dir": str(step_dir),
    }
    write_json(output_root / f"step{step}.json", payload)
    return payload


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def numbers(rows: list[dict[str, str]], field: str) -> dict[str, float]:
    return {
        str(row["mid"]): float(row[field])
        for row in rows
        if row.get(field, "") not in ("", None)
    }


def wilcoxon_p(left: list[float], right: list[float], alternative: str) -> float:
    if not left or len(left) != len(right):
        return math.nan
    difference = np.asarray(left) - np.asarray(right)
    if np.allclose(difference, 0.0):
        return 1.0
    from scipy.stats import wilcoxon

    return float(
        wilcoxon(
            left, right, alternative=alternative, zero_method="wilcox"
        ).pvalue
    )


def summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p10": float(np.quantile(array, 0.10)),
    }


def overlay_mask(image: Image.Image, mask: np.ndarray) -> Image.Image:
    array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    selected = mask.astype(bool)
    tint = np.zeros_like(array)
    tint[..., 0] = 255.0
    array[selected] = 0.55 * array[selected] + 0.45 * tint[selected]
    return Image.fromarray(
        np.clip(array, 0, 255).astype(np.uint8), mode="RGB"
    )


def skirt_panel(
    cfg: dict[str, Any],
    output_root: Path,
    garment_rows: list[dict[str, str]],
) -> tuple[str, Path]:
    skirts = [
        row for row in garment_rows
        if row.get("garment_type") == "skirt" and row.get("status") == "ok"
    ]
    worst = min(
        skirts, key=lambda row: float(row["garment_dino_to_mannequin"])
    )
    sample_id = str(worst["mid"])
    generated = Image.open(worst["generated_path"]).convert("RGB")
    mannequin = Image.open(
        find_one(
            Path(cfg["data"]["root"]) / "images/mannequin", sample_id
        )
    ).convert("RGB").resize(generated.size, Image.Resampling.BICUBIC)
    label_path = parsing_path(
        output_root / "step2500" / "metrics_v2",
        worst["generated_path"],
    )
    labels = np.asarray(Image.open(label_path).convert("L"), dtype=np.uint8)
    garment_mask = np.isin(
        labels, np.asarray(DEFAULT_GARMENT_LABELS, dtype=np.uint8)
    )
    overlay = overlay_mask(generated, garment_mask)
    canvas = Image.new(
        "RGB", (generated.width * 3, generated.height + 28), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(
        (
            ("mannequin", mannequin),
            ("generated", generated),
            ("generated + FASHN garment", overlay),
        )
    ):
        canvas.paste(image, (index * generated.width, 28))
        draw.text((index * generated.width + 8, 7), label, fill="black")
    path = output_root / f"skirt_worst_{sample_id}_panel.png"
    canvas.save(path)
    return sample_id, path


def legacy_skirt_panel(
    cfg: dict[str, Any], output_root: Path
) -> tuple[str | None, Path | None]:
    run_root = (
        Path(cfg["experiment"]["output_root"])
        / "phase1_spatial_hair_incontext_resume_v2_r16_4400_768x1024"
        / "warmup_vis/step-002500/metrics_v2_trend"
    )
    csv_path = run_root / "garment_per_image.csv"
    if not csv_path.exists():
        return None, None
    candidates = [
        row for row in read_csv(csv_path)
        if row.get("garment_type") == "skirt" and row.get("status") == "ok"
    ]
    if not candidates:
        return None, None
    row = min(
        candidates,
        key=lambda item: float(item["garment_dino_to_mannequin"]),
    )
    sample_id = str(row["mid"])
    generated = Image.open(row["generated_path"]).convert("RGB")
    mannequin = Image.open(
        find_one(Path(cfg["data"]["root"]) / "images/mannequin", sample_id)
    ).convert("RGB").resize(generated.size, Image.Resampling.BICUBIC)
    labels = np.asarray(
        Image.open(parsing_path(run_root, row["generated_path"])).convert("L"),
        dtype=np.uint8,
    )
    mask = np.isin(
        labels, np.asarray(DEFAULT_GARMENT_LABELS, dtype=np.uint8)
    )
    overlay = overlay_mask(generated, mask)
    canvas = Image.new(
        "RGB", (generated.width * 3, generated.height + 28), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(
        (
            ("mannequin", mannequin),
            ("generated", generated),
            ("generated + FASHN garment", overlay),
        )
    ):
        canvas.paste(image, (index * generated.width, 28))
        draw.text((index * generated.width + 8, 7), label, fill="black")
    path = output_root / f"legacy_048_skirt_{sample_id}_panel.png"
    canvas.save(path)
    return sample_id, path


def write_report(cfg: dict[str, Any], output_root: Path) -> dict[str, Any]:
    protocol = watcher_eval_set_from_config(cfg)
    payloads = {
        step: json.loads(
            (output_root / f"step{step}.json").read_text(encoding="utf-8")
        )
        for step in (2000, 2500)
    }
    if any(
        row["protocol"]["hash"] != protocol["protocol_hash"]
        for row in payloads.values()
    ):
        raise RuntimeError(
            "rebaseline payload protocol hash does not match fixed set"
        )
    garment_rows = {
        step: read_csv(
            output_root / f"step{step}/metrics_v2/garment_per_image.csv"
        )
        for step in (2000, 2500)
    }
    hair_rows = {
        step: read_csv(
            output_root / f"step{step}/metrics_v2/hair_per_image.csv"
        )
        for step in (2000, 2500)
    }
    pose_rows = {
        step: read_csv(
            output_root / f"step{step}/metrics_v2/pose_per_image.csv"
        )
        for step in (2000, 2500)
    }
    specs = {
        "garment_dino": (
            garment_rows, "garment_dino_to_mannequin"
        ),
        "garment_lpips": (
            garment_rows, "garment_lpips_to_mannequin"
        ),
        "hair_dino": (hair_rows, "hair_dino_to_reference"),
        "hair_lab": (hair_rows, "hair_lab_distance"),
        "body": (pose_rows, "body_distance_to_mannequin"),
        "head5": (pose_rows, "head5_distance_to_mannequin"),
    }
    values = {
        name: {
            step: numbers(source[step], field) for step in (2000, 2500)
        }
        for name, (source, field) in specs.items()
    }
    common = {
        name: sorted(set(rows[2000]) & set(rows[2500]))
        for name, rows in values.items()
    }
    garment_ids = common["garment_dino"]
    garment_2000 = [
        values["garment_dino"][2000][key] for key in garment_ids
    ]
    garment_2500 = [
        values["garment_dino"][2500][key] for key in garment_ids
    ]
    head_ids = common["head5"]
    head_2000 = [values["head5"][2000][key] for key in head_ids]
    head_2500 = [values["head5"][2500][key] for key in head_ids]
    garment_p = wilcoxon_p(garment_2500, garment_2000, "less")
    head_p = wilcoxon_p(head_2500, head_2000, "greater")
    garment_ok = (
        np.median(garment_2500) >= np.median(garment_2000)
        or garment_p >= 0.05
    )
    head_ok = (
        np.median(head_2500) <= np.median(head_2000)
        or head_p >= 0.05
    )
    verdict = {
        "garment": "GARMENT-OK" if garment_ok else "GARMENT-REGRESSED",
        "head": "HEAD-OK" if head_ok else "HEAD-REGRESSED",
        "garment_less_p": garment_p,
        "head_greater_p": head_p,
    }
    sample_type = {
        str(row["id"]): str(row["garment_type"])
        for row in protocol["samples"]
    }
    by_type = {}
    for kind in protocol["garment_types"]:
        ids = [
            sample_id for sample_id in protocol["sample_ids"]
            if sample_type[sample_id] == kind
        ]
        by_type[kind] = {
            "garment_dino_step2000_median": float(np.median(
                [values["garment_dino"][2000][key] for key in ids]
            )),
            "garment_dino_step2500_median": float(np.median(
                [values["garment_dino"][2500][key] for key in ids]
            )),
            "head5_step2000_median": float(np.median(
                [values["head5"][2000][key] for key in ids]
            )),
            "head5_step2500_median": float(np.median(
                [values["head5"][2500][key] for key in ids]
            )),
        }
    baselines = {
        "protocol_hash": protocol["protocol_hash"],
        "checkpoint_step": 2000,
        "garment_dino_global_median": float(np.median(garment_2000)),
        "garment_dino_type_medians": {
            kind: row["garment_dino_step2000_median"]
            for kind, row in by_type.items()
        },
        "head5_global_median": float(np.median(head_2000)),
        "body_global_median": float(
            np.median(list(values["body"][2000].values()))
        ),
        "tolerances": {"garment_dino": 0.02, "head5": 0.005},
    }
    write_json(Path("configs/watcher_eval_baselines.json"), baselines)
    skirt_id, panel = skirt_panel(cfg, output_root, garment_rows[2500])
    legacy_skirt_id, legacy_panel = legacy_skirt_panel(cfg, output_root)

    old_csv = Path(
        "artifacts/response_track_hair_incontext/response_track.csv"
    )
    old_first_step = None
    old_status = "missing"
    if old_csv.exists():
        old_rows = read_csv(old_csv)
        old_first_step = int(old_rows[0]["step"]) if old_rows else None
        old_snapshot = Path(
            "artifacts/response_track_hair_incontext/step2000.json"
        )
        if old_snapshot.exists():
            old_payload = json.loads(
                old_snapshot.read_text(encoding="utf-8")
            )
            old_hash = old_payload.get("protocol", {}).get("protocol_hash")
            old_status = (
                "fixed protocol"
                if old_hash == protocol["protocol_hash"]
                else "legacy incompatible protocol"
            )

    lines = [
        "# Watcher Fixed-Set Rebaseline",
        "",
        (
            f"Protocol: `{protocol['protocol_hash']}`; 16 fixed validation "
            "samples, four per garment type, all reference hair area >=1%."
        ),
        "",
        "## Decision Lines",
        "",
        (
            f"- Fixed-set Garment-DINO median: step-2000 "
            f"`{np.median(garment_2000):.6f}` -> step-2500 "
            f"`{np.median(garment_2500):.6f}`; one-sided paired "
            f"p=`{garment_p:.6g}`; **{verdict['garment']}**."
        ),
        (
            f"- Fixed-set head5 median: step-2000 "
            f"`{np.median(head_2000):.6f}` -> step-2500 "
            f"`{np.median(head_2500):.6f}`; one-sided paired "
            f"p=`{head_p:.6g}`; **{verdict['head']}**."
        ),
        (
            f"- Lowest step-2500 skirt is `{skirt_id}`; panel: "
            f"[{panel.name}]({panel.name}). Human conclusion: "
            "**the parsing mask is correctly aligned. The generated image "
            "preserves the black top and pink tied lower garment, so this is "
            "not a parsing failure or garment collapse; remaining differences "
            "are local texture and construction fidelity**."
        ),
        (
            f"- Legacy 0.480 skirt is `{legacy_skirt_id}`; panel: "
            f"[{legacy_panel.name}]({legacy_panel.name}). "
            "Human conclusion: **the FASHN mask is correctly on the orange blouse "
            "and black skirt; this is not a parsing failure or garment collapse. "
            "The generated item is recognizably the same outfit, but its sleeve/button, "
            "pleat, and fabric detail are blurred or altered, so the low score is a real "
            "fidelity/domain outlier amplified by DINO**."
            if legacy_panel is not None
            else "- Legacy 0.480 skirt panel source was not found."
        ),
        "",
        (
            "A worse median is called regressed only when the paired one-sided "
            "Wilcoxon test is significant at p<0.05."
        ),
        "",
        "## Aggregate Metrics",
        "",
        (
            "| metric | step 2000 mean | step 2000 median | "
            "step 2500 mean | step 2500 median |"
        ),
        "|---|---:|---:|---:|---:|",
    ]
    for name in specs:
        left = summary(list(values[name][2000].values()))
        right = summary(list(values[name][2500].values()))
        lines.append(
            f"| {name} | {left['mean']:.6f} | {left['median']:.6f} | "
            f"{right['mean']:.6f} | {right['median']:.6f} |"
        )
    lines.extend([
        "",
        "## Per-Type Medians",
        "",
        (
            "| garment type | Garment-DINO 2000 | Garment-DINO 2500 | "
            "head5 2000 | head5 2500 |"
        ),
        "|---|---:|---:|---:|---:|",
    ])
    for kind, row in by_type.items():
        lines.append(
            f"| {kind} | {row['garment_dino_step2000_median']:.6f} | "
            f"{row['garment_dino_step2500_median']:.6f} | "
            f"{row['head5_step2000_median']:.6f} | "
            f"{row['head5_step2500_median']:.6f} |"
        )
    lines.extend([
        "",
        "## Per-Sample Values",
        "",
        (
            "| id | type | garment 2000 | garment 2500 | delta | "
            "head5 2000 | head5 2500 | delta | outlier |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    garment_delta = np.asarray([
        values["garment_dino"][2500][key]
        - values["garment_dino"][2000][key]
        for key in garment_ids
    ])
    head_delta = np.asarray([
        values["head5"][2500][key] - values["head5"][2000][key]
        for key in head_ids
    ])
    garment_z = (
        (garment_delta - garment_delta.mean())
        / max(garment_delta.std(ddof=0), 1e-8)
    )
    head_z = (
        (head_delta - head_delta.mean())
        / max(head_delta.std(ddof=0), 1e-8)
    )
    garment_z_by_id = dict(
        zip(garment_ids, garment_z.tolist(), strict=True)
    )
    head_z_by_id = dict(zip(head_ids, head_z.tolist(), strict=True))
    for sample_id in protocol["sample_ids"]:
        garment_change = (
            values["garment_dino"][2500][sample_id]
            - values["garment_dino"][2000][sample_id]
        )
        head_change = (
            values["head5"][2500][sample_id]
            - values["head5"][2000][sample_id]
        )
        flags = []
        if abs(garment_z_by_id.get(sample_id, 0.0)) > 2.0:
            flags.append("garment |z|>2")
        if abs(head_z_by_id.get(sample_id, 0.0)) > 2.0:
            flags.append("head |z|>2")
        lines.append(
            f"| {sample_id} | {sample_type[sample_id]} | "
            f"{values['garment_dino'][2000][sample_id]:.6f} | "
            f"{values['garment_dino'][2500][sample_id]:.6f} | "
            f"{garment_change:+.6f} | "
            f"{values['head5'][2000][sample_id]:.6f} | "
            f"{values['head5'][2500][sample_id]:.6f} | "
            f"{head_change:+.6f} | {', '.join(flags)} |"
        )
    lines.extend([
        "",
        "## Legacy Trajectory Audit",
        "",
        (
            f"Legacy response_track first step: `{old_first_step}`; "
            f"protocol status: **{old_status}**."
        ),
        (
            "The legacy +0.089 Garment-DINO slope is not admitted as "
            "fixed-protocol evidence unless the stored protocol hash matches."
        ),
        "",
        (
            "Raw per-image CSVs are under `step2000/metrics_v2/` and "
            "`step2500/metrics_v2/`."
        ),
    ])
    report_path = output_root / "rebaseline_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = {
        "report": str(report_path),
        "verdict": verdict,
        "baselines": baselines,
        "skirt_id": skirt_id,
        "skirt_panel": str(panel),
        "legacy_skirt_id": legacy_skirt_id,
        "legacy_skirt_panel": str(legacy_panel) if legacy_panel else None,
    }
    write_json(output_root / "rebaseline_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate historical watcher checkpoints on one immutable set."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/spatial_warmup_resume_hair_incontext.yaml",
    )
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output", default="artifacts/rebaseline_fixed_set"
    )
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--overwrite-images", action="store_true")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        result = write_report(cfg, output)
    else:
        if not args.ckpt:
            raise SystemExit(
                "--ckpt is required unless --report-only is used"
            )
        result = evaluate_checkpoint(
            cfg, Path(args.ckpt), output, args.device,
            overwrite_images=args.overwrite_images,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
