from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import yaml

from source_ownership_ontology import ProtocolError, evaluate_hair_gate


HAIR_GATE_FAMILIES = {"identity_morphology", "hair_appearance_geometry", "occlusion"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def polygon_mask(raw: str, width: int, height: int) -> np.ndarray:
    points = json.loads(raw)
    if not isinstance(points, list) or len(points) < 3:
        raise ProtocolError("boundary_polygon_json must contain at least three [x,y] points")
    polygon = []
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            raise ProtocolError("each polygon point must be [x,y]")
        x, y = int(point[0]), int(point[1])
        if not 0 <= x < width or not 0 <= y < height:
            raise ProtocolError(f"polygon point {(x, y)} outside {width}x{height}")
        polygon.append((x, y))
    canvas = Image.new("1", (width, height), 0)
    ImageDraw.Draw(canvas).polygon(polygon, fill=1)
    return np.asarray(canvas, dtype=bool)


def majority(values: list[str]) -> str:
    counts = {value: values.count(value) for value in set(values)}
    best = max(counts.values())
    winners = sorted(value for value, count in counts.items() if count == best)
    if len(winners) != 1:
        raise ProtocolError(f"annotation tie requires adjudication: {winners}")
    return winners[0]


def frozen_face_only_policy(path: Path | None) -> bool:
    if path is None:
        return False
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return bool(
        isinstance(payload, dict)
        and payload.get("status") == "frozen"
        and payload.get("primary_policy") == "face_only"
        and payload.get("face_only_policy", {}).get("unsupported_hair") == "S_U"
        and payload.get("face_only_policy", {}).get("inner_face") == "S_I"
        and payload.get("face_only_policy", {}).get("face_head_boundary") == "S_X"
        and payload.get("face_only_policy", {}).get("mixing_with_main_hair_in_primary_test") == "forbidden"
    )


def evaluate(
    annotation_paths: list[Path],
    adjudication_path: Path,
    guideline_revision: int,
    fallback_config: Path | None = None,
) -> dict[str, Any]:
    if len(annotation_paths) != 3 or len(set(annotation_paths)) != 3:
        raise ProtocolError("exactly three distinct annotation files are required")
    support: dict[str, list[str]] = defaultdict(list)
    boundaries: dict[str, list[np.ndarray]] = defaultdict(list)
    occlusion: dict[str, list[str]] = defaultdict(list)
    seen_annotators: set[str] = set()
    expected_units: set[str] | None = None

    for path in annotation_paths:
        rows = read_rows(path)
        units = {row["unit_id"] for row in rows}
        if expected_units is None:
            expected_units = units
        elif units != expected_units:
            raise ProtocolError("all annotators must label the same units")
        annotators = {row.get("annotator_id", "").strip() for row in rows}
        if len(annotators) != 1 or "" in annotators:
            raise ProtocolError(f"{path} must contain one non-empty annotator_id")
        annotator = next(iter(annotators))
        if annotator in seen_annotators:
            raise ProtocolError(f"duplicate annotator_id {annotator!r}")
        seen_annotators.add(annotator)
        for row in rows:
            unit_id = row["unit_id"].strip()
            label = row.get("support_label", "").strip()
            if not label:
                raise ProtocolError(f"{path}: missing support_label for {unit_id}")
            if not row.get("owner_observable", "").strip():
                raise ProtocolError(f"{path}: missing owner_observable for {unit_id}")
            width = int(row["canvas_width"])
            height = int(row["canvas_height"])
            family = row.get("feature_family")
            if family in HAIR_GATE_FAMILIES:
                support[unit_id].append(label)
                boundaries[unit_id].append(polygon_mask(row["boundary_polygon_json"], width, height))
            if family == "occlusion":
                order = row.get("occlusion_order", "").strip()
                if not order:
                    raise ProtocolError(f"{path}: missing occlusion_order for {unit_id}")
                occlusion[unit_id].append(order)

    adjudicated = read_rows(adjudication_path)
    if {row["unit_id"] for row in adjudicated} != expected_units:
        raise ProtocolError("adjudication must cover exactly the raw annotation units")
    weights_by_image: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for row in adjudicated:
        unit_id = row["unit_id"]
        support_label = row.get("support_label", "").strip()
        if not support_label:
            support_label = majority(support[unit_id])
        if row.get("feature_family") in {"identity_morphology", "hair_appearance_geometry"}:
            try:
                weight = float(row.get("unit_weight", ""))
            except ValueError as exc:
                raise ProtocolError(f"adjudicated unit_weight must be numeric for {unit_id}") from exc
            if weight <= 0:
                raise ProtocolError(f"adjudicated unit_weight must be positive for {unit_id}")
            weights_by_image[row["image_id"]].append((weight, support_label))

    su_fraction_by_image = {}
    for image_id, values in weights_by_image.items():
        total = sum(weight for weight, _ in values)
        su_fraction_by_image[image_id] = sum(weight for weight, label in values if label == "S_U") / total

    result = evaluate_hair_gate(
        support_ratings_by_unit=support,
        boundary_masks_by_unit=boundaries,
        occlusion_ratings_by_unit=occlusion,
        su_fraction_by_image=su_fraction_by_image,
        guideline_revision=guideline_revision,
        fallback_policy_frozen=frozen_face_only_policy(fallback_config),
    )
    result["input_provenance"] = {
        "annotations": [{"path": str(path), "sha256": sha256_file(path)} for path in annotation_paths],
        "adjudication": {"path": str(adjudication_path), "sha256": sha256_file(adjudication_path)},
    }
    if fallback_config is not None:
        result["input_provenance"]["fallback_config"] = {
            "path": str(fallback_config),
            "sha256": sha256_file(fallback_config),
        }
    result["unit_count"] = len(expected_units or ())
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the frozen G2 ownership/hair policy gate.")
    parser.add_argument("--annotations", type=Path, nargs=3, required=True)
    parser.add_argument("--adjudication", type=Path, required=True)
    parser.add_argument("--guideline-revision", type=int, default=0)
    parser.add_argument("--fallback-config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate(args.annotations, args.adjudication, args.guideline_revision, args.fallback_config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
