from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from PIL import Image

from .validate import load_manifest


def _safe_link(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"missing layout source: {source}")
    source = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() == source:
            return
        destination.unlink()
    elif destination.exists():
        raise FileExistsError(f"refusing to replace non-symlink: {destination}")
    destination.symlink_to(source)


def _save_inverted_mask(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    values = np.asarray(Image.open(source).convert("L"), dtype=np.uint8)
    Image.fromarray(255 - values, mode="L").save(destination)


def _save_ita_replace_mask(replace_source: Path, face_source: Path, destination: Path, pad_x_fraction: float, pad_y_fraction: float) -> None:
    """Write an ITA-specific mask with room for a non-mannequin head shape."""
    if pad_x_fraction < 0.0 or pad_y_fraction < 0.0:
        raise ValueError("ITA head-box padding fractions must be non-negative")
    replace = np.asarray(Image.open(replace_source).convert("L"), dtype=np.uint8)
    face = np.asarray(Image.open(face_source).convert("L"), dtype=np.uint8)
    if face.shape != replace.shape:
        raise ValueError(f"ITA face/mask shape mismatch: {face.shape} vs {replace.shape}")
    expanded = np.where(replace >= 128, 255, 0).astype(np.uint8)
    ys, xs = np.where(face >= 128)
    if len(xs):
        pad_x = int(round(replace.shape[1] * pad_x_fraction))
        pad_y = int(round(replace.shape[0] * pad_y_fraction))
        x0 = max(0, int(xs.min()) - pad_x)
        x1 = min(replace.shape[1], int(xs.max()) + 1 + pad_x)
        y0 = max(0, int(ys.min()) - pad_y)
        y1 = min(replace.shape[0], int(ys.max()) + 1 + pad_y)
        expanded[y0:y1, x0:x1] = 255
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        destination.unlink()
    elif destination.exists() and not destination.is_file():
        raise FileExistsError(f"refusing to replace non-file: {destination}")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    Image.fromarray(expanded, mode="L").save(temporary, format="PNG")
    temporary.replace(destination)


def _phase(split: str) -> str:
    return "train" if split == "train" else "test"


def _rows(manifest_path: Path) -> list[dict[str, object]]:
    records = load_manifest(manifest_path)
    if not records:
        raise ValueError(f"empty manifest: {manifest_path}")
    return records


def materialize_refton(manifest_path: Path, prepared_root: Path, output_root: Path) -> Path:
    records = _rows(manifest_path)
    phase = _phase(str(records[0]["split"]))
    root = output_root / "refton"
    pair_lines: list[str] = []
    for record in records:
        name = f"{record.get('sample_name', record['mid'])}.png"
        base = root / phase
        mapping = {
            "image": "target",
            "person": "mannequin",
            "agnostic": "agnostic",
            "cloth": "face",
            "image_ref": "identity_card",
            "dense": "pose",
        }
        for directory, field in mapping.items():
            _safe_link(prepared_root / str(record[field]), base / directory / name)
        _safe_link(prepared_root / str(record["replace_mask"]), base / "agnostic_mask" / f"{Path(name).stem}_mask.png")
        pair_lines.append(f"{name} {name}\n")
    (root / phase / f"{phase}_pairs.txt").write_text("".join(pair_lines), encoding="utf-8")
    return root


def materialize_ita(manifest_path: Path, prepared_root: Path, output_root: Path, head_pad_x_fraction: float = 0.0625, head_pad_y_fraction: float = 0.0625) -> Path:
    records = _rows(manifest_path)
    phase = _phase(str(records[0]["split"]))
    root = output_root / "ita_mdt" / "zalando-hd-resized"
    pair_lines: list[str] = []
    for record in records:
        name = f"{record.get('sample_name', record['mid'])}.png"
        base = root / phase
        mapping = {
            "image": "target",
            "agnostic-v3.2": "agnostic",
            "cloth_sr": "face",
        }
        for directory, field in mapping.items():
            _safe_link(prepared_root / str(record[field]), base / directory / name)
        identity_person = (
            prepared_root
            / str(record["resolution"])
            / str(record["split"])
            / "identity_person"
            / f"{record['jid']}.png"
        )
        _safe_link(identity_person, base / "cloth" / name)
        pose_dense = (
            prepared_root
            / str(record["resolution"])
            / str(record["split"])
            / "pose_dense"
            / f"{record['mid']}.png"
        )
        _safe_link(pose_dense, base / "image-densepose" / name)
        # The loader inverts this M_replace mask. The wider head box lets a
        # counterfactual identity form its own head/hair silhouette.
        _save_ita_replace_mask(
            prepared_root / str(record["replace_mask"]),
            prepared_root / str(record["face_region"]),
            base / "agnostic-mask" / f"{Path(name).stem}_mask.png",
            head_pad_x_fraction,
            head_pad_y_fraction,
        )
        pair_lines.append(f"{name} {name}\n")
    (root / f"{phase}_pairs.txt").write_text("".join(pair_lines), encoding="utf-8")
    return root.parent


def _tagged_json(records: Iterable[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    return {
        "m2h": [
            {
                "file_name": f"{record.get('sample_name', record['mid'])}.png",
                "tag_info": [
                    {"tag_name": "item", "tag_category": "person identity"},
                    {"tag_name": "sleeveLength", "tag_category": None},
                    {"tag_name": "neckLine", "tag_category": None},
                ],
            }
            for record in records
        ]
    }


def materialize_idm(manifest_path: Path, prepared_root: Path, output_root: Path) -> Path:
    records = _rows(manifest_path)
    phase = _phase(str(records[0]["split"]))
    root = output_root / "idm_vton"
    pair_lines: list[str] = []
    for record in records:
        name = f"{record.get('sample_name', record['mid'])}.png"
        base = root / phase
        mapping = {
            "image": "target",
            "person": "mannequin",
            "cloth": "identity_card",
            "image-densepose": "pose",
        }
        for directory, field in mapping.items():
            _safe_link(prepared_root / str(record[field]), base / directory / name)
        _safe_link(prepared_root / str(record["replace_mask"]), base / "agnostic-mask" / f"{Path(name).stem}_mask.png")
        pair_lines.append(f"{name} {name}\n")
    (root / f"{phase}_pairs.txt").write_text("".join(pair_lines), encoding="utf-8")
    tagged = root / phase / f"vitonhd_{phase}_tagged.json"
    tagged.parent.mkdir(parents=True, exist_ok=True)
    tagged.write_text(json.dumps(_tagged_json(records), indent=2) + "\n", encoding="utf-8")
    return root


def materialize_mcld(manifest_path: Path, prepared_root: Path, output_root: Path) -> Path:
    records = _rows(manifest_path)
    phase = _phase(str(records[0]["split"]))
    root = output_root / "mcld"
    csv_rows: list[tuple[str, str]] = []
    for record in records:
        token = str(record.get("sample_name", record["mid"]))
        identity_name = f"{token}_identity.png"
        target_name = f"{token}_target.png"
        _safe_link(prepared_root / str(record["identity_card"]), root / phase / identity_name)
        _safe_link(prepared_root / str(record["target"]), root / phase / target_name)
        _safe_link(prepared_root / str(record["garment"]), root / f"{phase}_texture" / identity_name)
        _safe_link(prepared_root / str(record["pose"]), root / f"{phase}_densepose" / target_name)
        _safe_link(prepared_root / str(record["face_region"]), root / f"{phase}_face_mask" / target_name)
        _safe_link(prepared_root / str(record["face"]), root / f"{phase}_face_inputs" / identity_name)
        csv_rows.append((identity_name, target_name))
    csv_path = root / f"{phase}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("from", "to"))
        writer.writerows(csv_rows)
    return root


MATERIALIZERS = {
    "refton": materialize_refton,
    "ita_mdt": materialize_ita,
    "idm_vton": materialize_idm,
    "mcld": materialize_mcld,
}


def materialize(method: str, manifest_path: Path, prepared_root: Path, output_root: Path, config: Mapping[str, object] | None = None) -> Path:
    if method not in MATERIALIZERS:
        raise ValueError(f"unknown materializer {method}; choices={sorted(MATERIALIZERS)}")
    if method == "ita_mdt":
        options = config or {}
        return materialize_ita(
            manifest_path,
            prepared_root,
            output_root,
            float(options.get("ita_head_box_pad_x_fraction", 0.0625)),
            float(options.get("ita_head_box_pad_y_fraction", 0.0625)),
        )
    return MATERIALIZERS[method](manifest_path, prepared_root, output_root)
