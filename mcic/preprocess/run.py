from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from mcic.data.pairs import discover_pairs
from mcic.models.identity import offline_identity_embedding
from mcic.preprocess.audit import audit_dataset
from mcic.preprocess.parsing import compose_training_masks, load_base_masks
from mcic.utils.config import add_config_args, load_config, seeded_split
from mcic.utils.image import mask_bbox


def preprocess_dataset(config: dict) -> Path:
    root = Path(config["data"]["root"])
    cache = root / config["data"].get("cache_dir", "cache_mcic")
    pairs, _ = discover_pairs(root, config["data"]["mannequin_dir"], config["data"]["human_dir"])
    if not pairs:
        raise RuntimeError("No paired images found. Check data.root and paired directory names.")
    audit_dataset(config)
    splits = seeded_split([pair.sample_id for pair in pairs], config)
    metadata_path = cache / "metadata.jsonl"
    (cache / "masks").mkdir(parents=True, exist_ok=True)
    (cache / "faces").mkdir(parents=True, exist_ok=True)
    (cache / "identity_embeddings").mkdir(parents=True, exist_ok=True)
    quality_pass_count = 0
    with metadata_path.open("w", encoding="utf-8") as metadata:
        for pair in pairs:
            with Image.open(pair.human_path) as human_raw, Image.open(pair.mannequin_path) as man_raw:
                human = human_raw.convert("RGB")
                mannequin = man_raw.convert("RGB")
                base = load_base_masks(human, pair.sample_id, config)
                masks = compose_training_masks(base, config)
                face_box = mask_bbox(masks["face"])
                quality_pass = bool(face_box and min(face_box[2] - face_box[0], face_box[3] - face_box[1]) >= config["identity"].get("min_face_size", 20))
                quality_pass_count += int(quality_pass)
                safe_id = pair.sample_id.replace("/", "__")
                paths = {}
                for key in ("paired_mask", "cf_mask", "cloth_safe_mask", "ambiguous_mask"):
                    target = cache / "masks" / f"{safe_id}_{key}.png"
                    masks[key].save(target)
                    paths[f"{key}_path"] = str(target)
                if face_box:
                    human.crop(face_box).save(cache / "faces" / f"{safe_id}.png")
                    embedding = offline_identity_embedding(human, face_box, config)
                else:
                    embedding = np.zeros(int(config["identity"].get("embedding_dim", 512)), dtype=np.float32)
                    face_box = (0, 0, 1, 1)
                embedding_path = cache / "identity_embeddings" / f"{safe_id}.npy"
                np.save(embedding_path, embedding.astype(np.float32))
                row = {
                    "sample_id": pair.sample_id,
                    "mannequin_path": str(pair.mannequin_path),
                    "human_path": str(pair.human_path),
                    "original_mannequin_size": list(mannequin.size),
                    "original_human_size": list(human.size),
                    **paths,
                    "face_box": list(face_box),
                    "identity_embedding_path": str(embedding_path),
                    "face_quality_pass": quality_pass,
                    "split": splits[pair.sample_id],
                    "parsing_backend": config["preprocess"].get("parsing_backend", "heuristic"),
                }
                metadata.write(json.dumps(row, ensure_ascii=False) + "\n")
                panel = mannequin.copy()
                overlay = Image.new("RGBA", human.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay)
                draw.bitmap((0, 0), masks["paired_mask"], fill=(240, 70, 70, 100))
                panel = Image.alpha_composite(human.convert("RGBA"), overlay).convert("RGB")
                panel.save(cache / "visual_checks" / f"mask_{safe_id}.jpg")
    audit_path = cache / "audit_report.json"
    with audit_path.open("r", encoding="utf-8") as handle:
        audit = json.load(handle)
    audit["face_quality_pass_count"] = quality_pass_count
    audit["face_quality_pass_rate"] = quality_pass_count / len(pairs)
    with audit_path.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)
    return metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MCIC masks and identity cache.")
    add_config_args(parser)
    args = parser.parse_args()
    path = preprocess_dataset(load_config(args.config, args.overrides))
    print(f"Metadata written to {path}")


if __name__ == "__main__":
    main()
