from __future__ import annotations

import argparse
import csv
import gc
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw

from conditions import choose_dtype, find_one, get_resolution, load_yaml, make_image_ids, make_text_ids, seed_everything
from eval_b2 import make_cf_batch
from eval_watcher import decode_tokens
from metrics.common import expected_rows, safe_mean
from metrics_v2.common import read_csv, resolve_path, write_json
from metrics_v2.garment import run_garment_metrics
from metrics_v2.identity import run_identity_metrics
from metrics_v2.parsing import build_generated_parsing
from metrics_v2.pose import run_pose_metrics
from spatial_conditions import (
    add_control_samples,
    encode_packed_condition,
    image_mask_to_token_weights,
    mask_control_samples,
    masked_garment_image,
)
from train_paired import WarmupFlowModel, load_checkpoint, load_components


def scale_tag(value: float) -> str:
    return f"tile_{float(value):.1f}".replace(".", "p")


def select_probe_subset(
    subset: dict[str, Any],
    mid_count: int,
    identities_per_mid: int,
    seeds: Sequence[int] = (0,),
) -> dict[str, Any]:
    selected_mids = [str(value) for value in subset["mannequins"][: int(mid_count)]]
    selected_set = set(selected_mids)
    counts: dict[str, int] = defaultdict(int)
    pairs = []
    for pair in subset["pairs"]:
        mid = str(pair["mannequin_id"])
        if mid not in selected_set or counts[mid] >= int(identities_per_mid):
            continue
        item = dict(pair)
        item["mannequin_id"] = mid
        item["identity_id"] = str(pair["identity_id"])
        item["seeds"] = [int(value) for value in seeds]
        pairs.append(item)
        counts[mid] += 1
    missing = {mid: identities_per_mid - counts[mid] for mid in selected_mids if counts[mid] < identities_per_mid}
    if missing:
        raise RuntimeError(f"probe subset lacks requested identity pairs: {missing}")
    garment_types = {mid: subset.get("garment_types", {}).get(mid, "unknown") for mid in selected_mids}
    return {
        "seed": int(subset.get("seed", 0)),
        "mannequins": selected_mids,
        "pairs": pairs,
        "garment_types": garment_types,
        "garment_type_counts": dict(
            sorted((kind, list(garment_types.values()).count(kind)) for kind in set(garment_types.values()))
        ),
        "source_subset": "eval/cf_subset.json",
        "selection": "first mids in frozen order, first identities in frozen pair order, seed 0",
    }


class SpatialMaskedFluxMultiControlNetModel:
    """A thin facade over diffusers FluxMultiControlNetModel with pre-merge residual masks."""

    def __init__(self, controlnet) -> None:
        from diffusers import FluxMultiControlNetModel

        self.model = FluxMultiControlNetModel([controlnet])

    @torch.no_grad()
    def __call__(
        self,
        *,
        hidden_states: torch.Tensor,
        controlnet_cond: Sequence[torch.Tensor],
        controlnet_mode: Sequence[torch.Tensor],
        conditioning_scale: Sequence[float],
        residual_masks: Sequence[torch.Tensor | None],
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        timestep: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        guidance: torch.Tensor,
    ) -> tuple[list[torch.Tensor] | None, list[torch.Tensor] | None]:
        lengths = {
            len(controlnet_cond), len(controlnet_mode), len(conditioning_scale), len(residual_masks)
        }
        if lengths != {len(controlnet_cond)}:
            raise ValueError("multi-control condition/mode/scale/mask lengths must match")
        if len(self.model.nets) != 1:
            raise RuntimeError("Union probe must hold exactly one shared ControlNet instance")
        controlnet = self.model.nets[0]
        merged_blocks = None
        merged_single = None
        for condition, mode, scale, residual_mask in zip(
            controlnet_cond, controlnet_mode, conditioning_scale, residual_masks, strict=True
        ):
            if mode.ndim == 1:
                mode = mode[:, None]
            output = controlnet(
                hidden_states=hidden_states,
                controlnet_cond=condition,
                controlnet_mode=mode,
                conditioning_scale=float(scale),
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance,
                return_dict=True,
            )
            blocks = mask_control_samples(output.controlnet_block_samples, residual_mask)
            single = mask_control_samples(output.controlnet_single_block_samples, residual_mask)
            merged_blocks = add_control_samples(merged_blocks, blocks)
            merged_single = add_control_samples(merged_single, single)
        return merged_blocks, merged_single


@torch.no_grad()
def generate_tile(
    model: WarmupFlowModel,
    multi_control: SpatialMaskedFluxMultiControlNetModel,
    batch: dict[str, torch.Tensor],
    tile_latents: torch.Tensor,
    tile_mask: torch.Tensor,
    *,
    steps: int,
    seed: int,
    pose_scale: float,
    tile_scale: float,
    pose_mode: int,
    tile_mode: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(int(seed))
    z = torch.randn(
        1,
        (model.height // 16) * (model.width // 16),
        64,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    prompt = batch["prompt_embeds"].to(device=device, dtype=dtype)
    pooled = batch["pooled_prompt_embeds"].to(device=device, dtype=dtype)
    if prompt.ndim == 2:
        prompt = prompt.unsqueeze(0)
    if pooled.ndim == 1:
        pooled = pooled.unsqueeze(0)
    cond_tokens = model._condition_tokens(
        prompt,
        batch["appearance"].to(device=device, dtype=dtype).unsqueeze(0),
        batch["garment"].to(device=device, dtype=dtype).unsqueeze(0),
        batch["head_pose"].to(device=device, dtype=dtype).unsqueeze(0),
    )
    pose = batch["pose_latents"].to(device=device, dtype=dtype).unsqueeze(0)
    tile_latents = tile_latents.to(device=device, dtype=dtype)
    tile_mask = tile_mask.to(device=device, dtype=dtype)
    pulid_embed = batch["pulid_id_embed"].to(device=device, dtype=dtype).unsqueeze(0)
    img_ids = make_image_ids(model.width, model.height, device, dtype)
    txt_ids = make_text_ids(prompt.shape[1], device, dtype)
    guidance = torch.full((1,), model.guidance_scale, device=device, dtype=dtype)
    modes = [
        torch.full((1,), int(pose_mode), device=device, dtype=torch.long),
        torch.full((1,), int(tile_mode), device=device, dtype=torch.long),
    ]
    for index in range(int(steps)):
        tau = torch.full((1,), 1.0 - index / int(steps), device=device, dtype=dtype)
        cn_samples = multi_control(
            hidden_states=z,
            controlnet_cond=[pose, tile_latents],
            controlnet_mode=modes,
            conditioning_scale=[float(pose_scale), float(tile_scale)],
            residual_masks=[None, tile_mask],
            encoder_hidden_states=prompt,
            pooled_projections=pooled,
            timestep=tau,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
        )
        velocity = model._transformer_forward(
            z,
            tau,
            cond_tokens,
            pulid_embed,
            cn_samples,
            pooled=pooled,
            img_ids=img_ids,
        )
        z = z - (1.0 / int(steps)) * velocity
    return z


def _probe_paths(cfg: dict[str, Any], output_override: str | None = None) -> tuple[Path, Path]:
    root = Path(cfg["data"]["root"])
    output = resolve_path(root, output_override or cfg["tile_probe"]["output_root"])
    return root, output


def _probe_subset(cfg: dict[str, Any], smoke: bool) -> dict[str, Any]:
    root = Path(cfg["data"]["root"])
    subset = json.loads(resolve_path(root, cfg["data"]["cf_subset"]).read_text(encoding="utf-8"))
    pcfg = cfg["tile_probe"]
    return select_probe_subset(
        subset,
        2 if smoke else int(pcfg["mid_count"]),
        1 if smoke else int(pcfg["identities_per_mid"]),
        pcfg.get("seeds", [0]),
    )


def _write_status(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["mid", "jid", "seed", "tile_scale", "path", "status", "error"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_generation(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    from diffusers import AutoencoderKL

    root, output = _probe_paths(cfg, args.output_root)
    subset = _probe_subset(cfg, args.smoke)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "subset.json", subset)
    scales = args.tile_scales or [float(value) for value in cfg["tile_probe"]["tile_scales"]]
    if args.smoke and args.tile_scales is None:
        scales = [0.6]
    device = torch.device(args.device)
    dtype = choose_dtype(cfg["model"]["precision"])
    seed_everything(int(cfg["experiment"]["seed"]))
    transformer, controlnet, vae, adapter, pulid, notes = load_components(cfg, device, dtype)
    if vae is None:
        vae = AutoencoderKL.from_pretrained(
            cfg["model"]["base"], subfolder="vae", torch_dtype=dtype, local_files_only=True
        ).requires_grad_(False).to(device=device, dtype=dtype).eval()
    model = WarmupFlowModel(transformer, controlnet, adapter, pulid, cfg)
    checkpoint = Path(cfg["tile_probe"]["checkpoint"])
    step = load_checkpoint(checkpoint, model)
    model.eval()
    multi_control = SpatialMaskedFluxMultiControlNetModel(controlnet)
    cache = root / cfg["data"]["cache_dir"] / "samples"
    text_cache = root / cfg["data"]["cache_dir"] / "text" / "prompt.npz"
    pairs = [pair for index, pair in enumerate(subset["pairs"]) if index % args.num_shards == args.shard_index]
    tile_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    statuses: list[dict[str, Any]] = []
    for pair in pairs:
        mid = str(pair["mannequin_id"])
        jid = str(pair["identity_id"])
        if mid not in tile_cache:
            garment_image, cloth_mask = masked_garment_image(
                root,
                mid,
                cfg["data"]["resolution"],
                int(cfg["tile_probe"].get("neutral_gray", 127)),
            )
            tile_cache[mid] = (
                encode_packed_condition(vae, garment_image, cfg["data"]["resolution"], device, dtype),
                image_mask_to_token_weights(cloth_mask, model.width, model.height),
            )
        tile_latents, tile_mask = tile_cache[mid]
        batch = make_cf_batch(cache, text_cache, mid, jid)
        for seed in pair["seeds"]:
            for scale in scales:
                target = output / "gen" / scale_tag(scale) / f"{mid}__id{jid}__seed{int(seed)}.png"
                record = {
                    "mid": mid,
                    "jid": jid,
                    "seed": int(seed),
                    "tile_scale": float(scale),
                    "path": str(target),
                    "status": "ok",
                    "error": "",
                }
                try:
                    if args.overwrite or not target.exists():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        tokens = generate_tile(
                            model,
                            multi_control,
                            batch,
                            tile_latents,
                            tile_mask,
                            steps=int(cfg["eval"]["generate_steps"]),
                            seed=int(seed),
                            pose_scale=float(cfg["tile_probe"]["pose_scale"]),
                            tile_scale=float(scale),
                            pose_mode=int(cfg["tile_probe"]["pose_mode"]),
                            tile_mode=int(cfg["tile_probe"]["tile_mode"]),
                            device=device,
                            dtype=dtype,
                        )
                        decode_tokens(vae, tokens, cfg["data"]["resolution"]).save(target)
                except Exception as exc:  # noqa: BLE001
                    record["status"] = "failed"
                    record["error"] = str(exc)
                statuses.append(record)
                print(json.dumps(record, ensure_ascii=False), flush=True)
    _write_status(output / "status" / f"shard_{args.shard_index:02d}.csv", statuses)
    write_json(
        output / "launch" / f"shard_{args.shard_index:02d}.json",
        {
            "checkpoint": str(checkpoint),
            "checkpoint_step": step,
            "device": str(device),
            "num_shards": int(args.num_shards),
            "tile_scales": scales,
            "pose_scale": float(cfg["tile_probe"]["pose_scale"]),
            "controlnet": notes["controlnet"],
            "multi_control": "FluxMultiControlNetModel with one shared Union net and pre-merge tile residual mask",
        },
    )


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_evaluation(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    root, output = _probe_paths(cfg, args.output_root)
    subset = _probe_subset(cfg, args.smoke)
    scales = args.tile_scales or [float(value) for value in cfg["tile_probe"]["tile_scales"]]
    if args.smoke and args.tile_scales is None:
        scales = [0.6]
    for scale in scales:
        gen_dir = output / "gen" / scale_tag(scale)
        metric_dir = output / "metrics" / scale_tag(scale)
        rows = expected_rows(cfg, subset, gen_dir)
        missing = [str(row["path"]) for row in rows if not row["path"].exists()]
        if missing:
            raise FileNotFoundError(f"tile scale={scale} missing {len(missing)} images; first={missing[:10]}")
        write_json(metric_dir / "parsing_summary.json", build_generated_parsing(cfg, rows, metric_dir, args.device))
        _cleanup_cuda()
        run_garment_metrics(cfg, rows, metric_dir, args.device)
        _cleanup_cuda()
        run_pose_metrics(cfg, rows, metric_dir, args.pose_device)
        run_identity_metrics(cfg, subset, gen_dir, metric_dir, args.device)
        _cleanup_cuda()


def _mean_field(path: Path, field: str) -> float | None:
    return safe_mean([row.get(field) for row in read_csv(path)])


def _face_rate(path: Path) -> float | None:
    rows = read_csv(path)
    return sum(row.get("status") == "ok" for row in rows) / len(rows) if rows else None


def _filter_baseline(path: Path, keys: set[tuple[str, str, int]]) -> list[dict[str, str]]:
    return [
        row for row in read_csv(path)
        if (str(row["mid"]), str(row["jid"]), int(row["seed"])) in keys
    ]


def _write_filtered_mean(rows: list[dict[str, str]], field: str) -> float | None:
    return safe_mean([row.get(field) for row in rows])


def make_panels(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    root, output = _probe_paths(cfg, args.output_root)
    subset = _probe_subset(cfg, args.smoke)
    scales = args.tile_scales or [float(value) for value in cfg["tile_probe"]["tile_scales"]]
    if args.smoke and args.tile_scales is None:
        scales = [0.6]
    width = int(cfg["tile_probe"].get("panel_tile_width", 192))
    height = int(cfg["tile_probe"].get("panel_tile_height", 256))
    label_h = 26
    columns = ["mannequin", "A4"] + [f"tile {value:.1f}" for value in scales]
    by_mid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in subset["pairs"]:
        by_mid[str(pair["mannequin_id"])].append(pair)
    panel_dir = output / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = resolve_path(root, cfg["tile_probe"]["baseline_gen_dir"])
    for mid, pairs in by_mid.items():
        canvas = Image.new("RGB", (width * len(columns), label_h + height * len(pairs)), "white")
        draw = ImageDraw.Draw(canvas)
        for column, label in enumerate(columns):
            draw.text((column * width + 6, 7), label, fill="black")
        mannequin = Image.open(find_one(root / "images/mannequin", mid)).convert("RGB").resize((width, height))
        for row_index, pair in enumerate(pairs):
            jid = str(pair["identity_id"])
            seed = int(pair["seeds"][0])
            paths = [baseline_dir / f"{mid}__id{jid}__seed{seed}.png"] + [
                output / "gen" / scale_tag(scale) / f"{mid}__id{jid}__seed{seed}.png" for scale in scales
            ]
            missing = [str(path) for path in paths if not path.exists()]
            if missing:
                raise FileNotFoundError(f"panel missing images: {missing}")
            y = label_h + row_index * height
            canvas.paste(mannequin, (0, y))
            for column, path in enumerate(paths, start=1):
                image = Image.open(path).convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
                canvas.paste(image, (column * width, y))
            draw.rectangle((0, y, width - 1, y + 22), fill=(255, 255, 255))
            draw.text((5, y + 5), f"mid {mid} / id {jid}", fill="black")
        canvas.save(panel_dir / f"panel_{mid}.png")


def write_report(args: argparse.Namespace, cfg: dict[str, Any]) -> Path:
    root, output = _probe_paths(cfg, args.output_root)
    subset = _probe_subset(cfg, args.smoke)
    scales = args.tile_scales or [float(value) for value in cfg["tile_probe"]["tile_scales"]]
    if args.smoke and args.tile_scales is None:
        scales = [0.6]
    keys = {
        (str(pair["mannequin_id"]), str(pair["identity_id"]), int(seed))
        for pair in subset["pairs"] for seed in pair["seeds"]
    }
    baseline_dir = resolve_path(root, cfg["tile_probe"]["baseline_metrics_dir"])
    baseline_garment = _filter_baseline(baseline_dir / "garment_per_image.csv", keys)
    baseline_pose = _filter_baseline(baseline_dir / "pose_per_image.csv", keys)
    baseline_identity = _filter_baseline(baseline_dir / "identity/deltaid_per_image.csv", keys)
    baseline = {
        "garment_dino": _write_filtered_mean(baseline_garment, "garment_dino_to_mannequin"),
        "garment_lpips": _write_filtered_mean(baseline_garment, "garment_lpips_to_mannequin"),
        "body_pose": _write_filtered_mean(baseline_pose, "body_distance_to_mannequin"),
        "head_pose": _write_filtered_mean(baseline_pose, "head5_distance_to_mannequin"),
        "face_rate": sum(row.get("status") == "ok" for row in baseline_identity) / len(baseline_identity),
    }
    rows = []
    for scale in scales:
        metric_dir = output / "metrics" / scale_tag(scale)
        identity_csv = metric_dir / "identity/deltaid_per_image.csv"
        item = {
            "scale": float(scale),
            "garment_dino": _mean_field(metric_dir / "garment_per_image.csv", "garment_dino_to_mannequin"),
            "garment_lpips": _mean_field(metric_dir / "garment_per_image.csv", "garment_lpips_to_mannequin"),
            "body_pose": _mean_field(metric_dir / "pose_per_image.csv", "body_distance_to_mannequin"),
            "head_pose": _mean_field(metric_dir / "pose_per_image.csv", "head5_distance_to_mannequin"),
            "face_rate": _face_rate(identity_csv),
        }
        item["garment_gain"] = item["garment_dino"] - baseline["garment_dino"]
        item["pose_not_worse"] = item["body_pose"] <= baseline["body_pose"]
        item["face_not_worse"] = item["face_rate"] >= baseline["face_rate"]
        rows.append(item)
    confirm = float(cfg["tile_probe"]["garment_gain_confirm"])
    reprobe = float(cfg["tile_probe"]["garment_gain_reprobe"])
    confirmed = [
        row for row in rows
        if row["garment_gain"] >= confirm and row["pose_not_worse"] and row["face_not_worse"]
    ]
    best = max(rows, key=lambda row: row["garment_gain"])
    if confirmed:
        decision = "CONFIRMED: spatial garment injection passes; proceed to in-context reference latents."
    elif best["garment_gain"] < reprobe:
        decision = "REPROBE: gain <0.02; audit tile masking/semantics, then test canny/depth once."
    else:
        decision = "INCONCLUSIVE: gain is between 0.02 and 0.05 or a pose/face guard failed; do not start Step 1 yet."
    lines = [
        "# Tile Multi-Control Zero-Shot Probe",
        "",
        f"**{decision}**",
        "",
        f"Frozen A4 checkpoint; {len(subset['mannequins'])} frozen mids x "
        f"{len(subset['pairs']) // max(1, len(subset['mannequins']))} identities x seed 0. "
        "One shared InstantX Union ControlNet runs pose mode 4 and tile mode 1. "
        "Tile residuals are soft-masked before addition.",
        "",
        "| Setting | Garment-DINO | gain vs A4 | Garment-LPIPS | body pose dist | head-5 dist | face detect | pose guard | face guard |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
        f"| Current A4 | {baseline['garment_dino']:.4f} | 0.0000 | {baseline['garment_lpips']:.4f} | "
        f"{baseline['body_pose']:.4f} | {baseline['head_pose']:.4f} | {baseline['face_rate']:.4f} | baseline | baseline |",
    ]
    for row in rows:
        lines.append(
            f"| tile {row['scale']:.1f} | {row['garment_dino']:.4f} | {row['garment_gain']:+.4f} | "
            f"{row['garment_lpips']:.4f} | {row['body_pose']:.4f} | {row['head_pose']:.4f} | "
            f"{row['face_rate']:.4f} | {row['pose_not_worse']} | {row['face_not_worse']} |"
        )
    lines += [
        "",
        "## Preregistered Decision",
        "",
        f"- Confirm spatial injection when Garment-DINO gain >= {confirm:.2f} and strict mean pose/face guards do not regress.",
        f"- If best gain < {reprobe:.2f}, inspect tile semantics/masking and run one canny/depth probe before rejecting the direction.",
        f"- Best observed scale: `{best['scale']:.1f}` with gain `{best['garment_gain']:+.4f}`.",
        "",
        "## Panels",
        "",
        f"Per-mid panels: `{output / 'panels'}`",
    ]
    path = output / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output / "report.json", {"decision": decision, "baseline": baseline, "scales": rows})
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-training masked tile multi-control probe.")
    parser.add_argument("--config", default="configs/tile_probe.yaml")
    parser.add_argument("--stage", choices=["generate", "evaluate", "panels", "report", "all"], default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pose-device", default="cuda:1")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--tile-scales", type=float, nargs="+", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require 0 <= shard-index < num-shards")
    cfg = load_yaml(args.config)
    if args.stage in {"generate", "all"}:
        run_generation(args, cfg)
    if args.stage in {"evaluate", "all"}:
        run_evaluation(args, cfg)
    if args.stage in {"panels", "all"}:
        make_panels(args, cfg)
    if args.stage in {"report", "all"}:
        print(write_report(args, cfg))


if __name__ == "__main__":
    main()
