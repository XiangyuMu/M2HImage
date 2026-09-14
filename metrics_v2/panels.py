from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from metrics_v2.common import deterministic_panel_cases, find_image, image_size, read_rgb, write_csv, write_json
from metrics_v2.parsing import load_generated_masks, reference_hair_mask, source_garment_mask


GREEN = np.asarray([30, 220, 90], dtype=np.float32)
BLUE = np.asarray([50, 140, 255], dtype=np.float32)


def overlay(image: np.ndarray, masks: list[tuple[np.ndarray, np.ndarray]], alpha: float = 0.42) -> np.ndarray:
    result = image.astype(np.float32).copy()
    for mask, color in masks:
        selected = mask.astype(bool)
        result[selected] = result[selected] * (1.0 - alpha) + color * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def tile(image: np.ndarray, label: str, width: int, height: int) -> Image.Image:
    canvas = Image.new("RGB", (width, height + 28), "white")
    source = Image.fromarray(image).convert("RGB")
    source.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas.paste(source, ((width - source.width) // 2, (height - source.height) // 2))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, height, width, height + 28), fill=(245, 245, 245))
    draw.text((8, height + 7), label, fill=(20, 20, 20), font=ImageFont.load_default())
    return canvas


def build_panels(
    cfg: dict[str, Any], subset: dict[str, Any], rows: list[dict[str, Any]], out_dir: str | Path, run_name: str
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    panel_dir = out_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    pcfg = cfg["metrics_v2"]["panels"]
    width = int(pcfg.get("tile_width", 288))
    height = int(pcfg.get("tile_height", 384))
    size = image_size(cfg)
    cases = deterministic_panel_cases(
        subset,
        seed=int(pcfg.get("seed", 20260817)),
        mid_count=int(pcfg.get("mid_count", 8)),
        identities_per_mid=int(pcfg.get("identities_per_mid", 2)),
    )
    row_map = {(str(row["mid"]), str(row["jid"]), int(row["seed"])): row for row in rows}
    root = Path(cfg["data"]["root"])
    index_rows: list[dict[str, Any]] = []
    panels: list[Path] = []
    for mid, jid, seed in cases:
        row = row_map.get((mid, jid, seed))
        if row is None:
            continue
        mannequin_path = find_image(root, "images/mannequin", mid)
        reference_path = find_image(root, "images/human", jid)
        mannequin = read_rgb(mannequin_path, size)
        reference = read_rgb(reference_path, size)
        generated = read_rgb(row["path"], size)
        source_cloth = source_garment_mask(cfg, mid, size)
        reference_hair = reference_hair_mask(cfg, jid, size)
        generated_cloth, generated_hair, mask_source = load_generated_masks(cfg, out_dir, row)
        images = [
            tile(overlay(mannequin, [(source_cloth, GREEN)]), "mannequin / cloth=green", width, height),
            tile(overlay(reference, [(reference_hair, BLUE)]), "reference / hair=blue", width, height),
            tile(
                overlay(generated, [(generated_cloth, GREEN), (generated_hair, BLUE)]),
                f"{run_name} / cloth+hair",
                width,
                height,
            ),
        ]
        panel = Image.new("RGB", (width * 3, height + 28), "white")
        for index, image in enumerate(images):
            panel.paste(image, (index * width, 0))
        panel_path = panel_dir / f"{mid}__id{jid}__seed{seed}.png"
        panel.save(panel_path)
        panels.append(panel_path)
        index_rows.append(
            {
                "run": run_name,
                "mid": mid,
                "jid": jid,
                "seed": seed,
                "garment_type": row.get("garment_type", "unknown"),
                "mask_source": mask_source,
                "panel": str(panel_path),
            }
        )
    write_csv(out_dir / "panel_index.csv", index_rows, ["run", "mid", "jid", "seed", "garment_type", "mask_source", "panel"])
    contact_path = out_dir / "qualitative_mask_overlays.png"
    if panels:
        thumbs = []
        for panel_path in panels:
            panel = Image.open(panel_path).convert("RGB")
            panel.thumbnail((648, 330), Image.Resampling.LANCZOS)
            thumbs.append(panel.copy())
        columns = 2
        cell_width = max(image.width for image in thumbs)
        cell_height = max(image.height for image in thumbs)
        contact = Image.new("RGB", (columns * cell_width, ((len(thumbs) + 1) // 2) * cell_height), "white")
        for index, image in enumerate(thumbs):
            contact.paste(image, ((index % columns) * cell_width, (index // columns) * cell_height))
        contact.save(contact_path)
    summary = {
        "seed": int(pcfg.get("seed", 20260817)),
        "case_count": len(index_rows),
        "mask_colors": {"garment": "green", "hair": "blue"},
        "contact_sheet": str(contact_path),
        "index_csv": str(out_dir / "panel_index.csv"),
    }
    write_json(out_dir / "panels_summary.json", summary)
    return summary
