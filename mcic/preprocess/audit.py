from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps, ImageDraw

from mcic.data.pairs import discover_pairs
from mcic.utils.config import add_config_args, load_config, write_json


def audit_dataset(config: dict) -> dict:
    root = Path(config["data"]["root"])
    cache = root / config["data"].get("cache_dir", "cache_mcic")
    pairs, failures = discover_pairs(root, config["data"]["mannequin_dir"], config["data"]["human_dir"])
    report = {
        "root": str(root),
        "paired_count": len(pairs),
        "missing_human": failures["missing_human"],
        "missing_mannequin": failures["missing_mannequin"],
        "invalid_images": [],
        "size_mismatches": [],
        "resolution_histogram": {},
    }
    valid_pairs = []
    for pair in pairs:
        try:
            with Image.open(pair.mannequin_path) as mannequin, Image.open(pair.human_path) as human:
                man_size, human_size = mannequin.size, human.size
                key = f"{human_size[0]}x{human_size[1]}"
                report["resolution_histogram"][key] = report["resolution_histogram"].get(key, 0) + 1
                if man_size != human_size:
                    report["size_mismatches"].append(pair.sample_id)
                valid_pairs.append(pair)
        except OSError as exc:
            report["invalid_images"].append({"sample_id": pair.sample_id, "error": str(exc)})
    preview_dir = cache / "visual_checks"
    preview_dir.mkdir(parents=True, exist_ok=True)
    limit = int(config.get("preprocess", {}).get("max_visual_checks", 200))
    for pair in valid_pairs[:limit]:
        with Image.open(pair.mannequin_path) as man, Image.open(pair.human_path) as human:
            man = ImageOps.fit(man.convert("RGB"), (256, 384))
            human = ImageOps.fit(human.convert("RGB"), (256, 384))
            canvas = Image.new("RGB", (512, 412), "white")
            canvas.paste(man, (0, 28))
            canvas.paste(human, (256, 28))
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 8), f"{pair.sample_id}: mannequin | human", fill="black")
            canvas.save(preview_dir / f"pair_{pair.sample_id.replace('/', '__')}.jpg")
    write_json(cache / "audit_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit MCIC paired image data.")
    add_config_args(parser)
    args = parser.parse_args()
    report = audit_dataset(load_config(args.config, args.overrides))
    print(f"Found {report['paired_count']} paired samples. Report written under cache_mcic.")


if __name__ == "__main__":
    main()
