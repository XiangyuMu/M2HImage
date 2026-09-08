"""CPU-only protocol revalidation for the frozen A4/B2-cont dev128 pairs.

This script is intentionally read-only with respect to the original dataset and
old protocol artifacts. It preserves the frozen 128 pairs exactly, hard-fails on
exact train-human SHA overlap for the dev128 target humans or references, and
writes a fresh sensitivity manifest plus near-duplicate scores.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
from PIL import Image

SEED = 20260907
SIZE = (384, 512)
CENTER_FRACTION = 0.5
SCHEMA = "m2h_minimal_revalidation_protocol_v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_key(value: str) -> str:
    return hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest()


def write_json(path: Path, value) -> None:
    with path.open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def write_csv(path: Path, fields: list[str], rows) -> None:
    with path.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_pairs(path: Path) -> dict:
    payload = json.loads(path.read_text())
    pairs = payload.get("pairs")
    require(isinstance(pairs, list) and len(pairs) == 128, "frozen dev128 must contain exactly 128 pairs")
    mids = defaultdict(list)
    for i, row in enumerate(pairs):
        require(isinstance(row, dict), "pair rows must be objects")
        require(isinstance(row.get("mid"), str) and row["mid"], "pair mid must be a nonempty string")
        require(isinstance(row.get("jid"), str) and row["jid"], "pair jid must be a nonempty string")
        mids[row["mid"]].append(row["jid"])
        row["_pair_index"] = i
    require(len(mids) == 64 and all(len(v) == 2 and len(set(v)) == 2 for v in mids.values()),
            "frozen dev128 must be 64 target M IDs with two distinct references each")
    for row in pairs:
        row.pop("_pair_index", None)
    return payload


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_per_image(path: Path) -> tuple[list[dict], dict[tuple[str, str], dict]]:
    rows: list[dict] = []
    keyed: dict[tuple[str, str], dict] = {}
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            key = (row["id"], row["role"])
            require(key not in keyed, f"duplicate per_image key: {key}")
            require(row.get("status") == "ok", f"non-ok image audit row: {key}")
            require(row["role"] in ("human", "mannequin"), f"unexpected role: {row['role']}")
            require(len(row.get("sha256", "")) == 64, f"malformed sha256 for {key}")
            row["_image_index"] = len(rows)
            keyed[key] = row
            rows.append(row)
    return rows, keyed


class UnionFind:
    def __init__(self, values: list[str]):
        self.values = values
        self.index = {v: i for i, v in enumerate(values)}
        self.parent = list(range(len(values)))
        self.size = [1] * len(values)

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union_value(self, a: str, b: str) -> None:
        i, j = self.index[a], self.index[b]
        i, j = self.find(i), self.find(j)
        if i == j:
            return
        if self.size[i] < self.size[j]:
            i, j = j, i
        self.parent[j] = i
        self.size[i] += self.size[j]

    def component_map(self) -> dict[str, str]:
        groups: dict[int, list[str]] = defaultdict(list)
        for value, i in self.index.items():
            groups[self.find(i)].append(value)
        result = {}
        for members in groups.values():
            cid = hashlib.sha256("\n".join(sorted(members)).encode()).hexdigest()
            for member in members:
                result[member] = cid
        return result


def exact_human_train_overlaps(exact_groups_path: Path, per_image: dict[tuple[str, str], dict]) -> list[dict]:
    groups = json.loads(exact_groups_path.read_text())
    overlaps: list[dict] = []
    for group_index, group in enumerate(groups):
        if group.get("role") != "human":
            continue
        members = group.get("members", [])
        train_members = [m for m in members if m.get("split") == "train"]
        if not train_members:
            continue
        for member in members:
            if member.get("split") == "train":
                continue
            sid = member["id"]
            row = per_image[(sid, "human")]
            overlaps.append({
                "group_index": group_index,
                "sha256": group["sha256"],
                "id": sid,
                "split": member["split"],
                "train_ids": "|".join(sorted(m["id"] for m in train_members)),
                "path": row["path"],
            })
    return sorted(overlaps, key=lambda r: (r["split"], r["id"], r["sha256"]))


def image_array(path: str | Path, size: tuple[int, int] = SIZE) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize(size, Image.Resampling.LANCZOS), dtype=np.float64)


def jpeg_roundtrip(arr: np.ndarray, quality: int) -> np.ndarray:
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    with Image.open(buf) as out:
        return np.asarray(out.convert("RGB"), dtype=np.float64)


def down_up(arr: np.ndarray) -> np.ndarray:
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
    small = im.resize((SIZE[0] // 2, SIZE[1] // 2), Image.Resampling.BICUBIC)
    return np.asarray(small.resize(SIZE, Image.Resampling.BICUBIC), dtype=np.float64)


def crop_center(arr: np.ndarray, fraction: float = CENTER_FRACTION) -> np.ndarray:
    h, w = arr.shape[:2]
    ch, cw = int(h * fraction), int(w * fraction)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return arr[y0:y0 + ch, x0:x0 + cw]


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    # Global SSIM over RGB channels. This is a deterministic CPU proxy, not a
    # semantic duplicate or identity detector.
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    vals = []
    for c in range(a.shape[2]):
        x, y = a[:, :, c], b[:, :, c]
        mux, muy = x.mean(), y.mean()
        vx, vy = x.var(), y.var()
        cov = ((x - mux) * (y - muy)).mean()
        vals.append(((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux ** 2 + muy ** 2 + c1) * (vx + vy + c2)))
    return float(np.mean(vals))


def metrics(left_path: str, right_path: str) -> dict[str, float]:
    a = image_array(left_path)
    b = image_array(right_path)
    return {
        "ssim_384x512": ssim(a, b),
        "center_ssim": ssim(crop_center(a), crop_center(b)),
        "mae_384x512": float(np.mean(np.abs(a - b))),
        "center_mae": float(np.mean(np.abs(crop_center(a) - crop_center(b)))),
    }


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "min": None, "median": None, "max": None}
    return {"n": len(values), "min": min(values), "median": statistics.median(values), "max": max(values)}


def calibrate(train_humans: list[dict], sample_n: int = 32) -> dict:
    eligible = sorted(train_humans, key=lambda r: stable_key(r["id"] + ":" + r["sha256"]))[:sample_n]
    require(len(eligible) == sample_n, "not enough train-only human images for fixed calibration sample")
    positive_rows = []
    negative_rows = []
    for row in eligible:
        arr = image_array(row["path"])
        for transform, other in (("jpeg_q85", jpeg_roundtrip(arr, 85)),
                                 ("jpeg_q95", jpeg_roundtrip(arr, 95)),
                                 ("down_up_half_bicubic", down_up(arr))):
            positive_rows.append({
                "kind": "positive",
                "transform": transform,
                "id": row["id"],
                "ssim_384x512": ssim(arr, other),
                "center_ssim": ssim(crop_center(arr), crop_center(other)),
                "mae_384x512": float(np.mean(np.abs(arr - other))),
                "center_mae": float(np.mean(np.abs(crop_center(arr) - crop_center(other)))),
            })
    for left, right in zip(eligible, eligible[1:] + eligible[:1]):
        m = metrics(left["path"], right["path"])
        negative_rows.append({"kind": "negative", "left_id": left["id"], "right_id": right["id"], **m})
    pos_ssim = [r["ssim_384x512"] for r in positive_rows]
    neg_ssim = [r["ssim_384x512"] for r in negative_rows]
    pos_center = [r["center_ssim"] for r in positive_rows]
    neg_center = [r["center_ssim"] for r in negative_rows]
    pos_mae = [r["mae_384x512"] for r in positive_rows]
    neg_mae = [r["mae_384x512"] for r in negative_rows]
    separates = min(pos_ssim) > max(neg_ssim) and min(pos_center) > max(neg_center) and max(pos_mae) < min(neg_mae)
    threshold = {
        "ssim_384x512_gte": (min(pos_ssim) + max(neg_ssim)) / 2 if separates else min(pos_ssim),
        "center_ssim_gte": (min(pos_center) + max(neg_center)) / 2 if separates else min(pos_center),
        "mae_384x512_lte": (max(pos_mae) + min(neg_mae)) / 2 if separates else max(pos_mae),
        "rule": "high_risk if SSIM and center SSIM are above thresholds and MAE is below threshold",
        "qualified": separates,
    }
    return {
        "sample_ids": [r["id"] for r in eligible],
        "positive_summary": {
            "ssim_384x512": summarize(pos_ssim),
            "center_ssim": summarize(pos_center),
            "mae_384x512": summarize(pos_mae),
        },
        "negative_summary": {
            "ssim_384x512": summarize(neg_ssim),
            "center_ssim": summarize(neg_center),
            "mae_384x512": summarize(neg_mae),
        },
        "false_positive_negatives_under_rule": sum(
            r["ssim_384x512"] >= threshold["ssim_384x512_gte"]
            and r["center_ssim"] >= threshold["center_ssim_gte"]
            and r["mae_384x512"] <= threshold["mae_384x512_lte"]
            for r in negative_rows
        ),
        "threshold": threshold,
        "positive_rows": positive_rows,
        "negative_rows": negative_rows,
        "limitations": [
            "Calibration uses synthetic JPEG/resize perturbations as positives, not manual labels.",
            "Different train images are proxy negatives and can contain true near duplicates.",
            "Scores are sensitivity evidence only and do not block the main revalidation once exact overlap is clear.",
        ],
    }


def high_risk(row: dict, threshold: dict) -> bool:
    return (
        float(row["ssim_384x512"]) >= float(threshold["ssim_384x512_gte"])
        and float(row["center_ssim"]) >= float(threshold["center_ssim_gte"])
        and float(row["mae_384x512"]) <= float(threshold["mae_384x512_lte"])
    )


def collect_near_candidates(image_pairs: Path, images: list[dict], dev_ids: set[str]) -> list[dict]:
    rows = []
    seen = set()
    with gzip.open(image_pairs, "rt", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            li, ri = int(raw["left_image_index"]), int(raw["right_image_index"])
            a, b = images[li], images[ri]
            for dev, train in ((a, b), (b, a)):
                if dev["id"] in dev_ids and dev["role"] == "human" and train["split"] == "train" and train["role"] == "human":
                    key = (dev["id"], train["id"])
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append({
                        "dev_id": dev["id"],
                        "dev_role": dev["role"],
                        "dev_path": dev["path"],
                        "train_id": train["id"],
                        "train_path": train["path"],
                        "dhash_hamming": raw["hamming"],
                        "sha256_equal": raw["sha256_equal"],
                    })
    return sorted(rows, key=lambda r: (r["dev_id"], int(r["dhash_hamming"]), r["train_id"]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old-protocol", type=Path, default=Path("/data/muxiangyu/pythonPrograms/M2HImage/artifacts/m2h_ab_objective_v1/data_protocol_v2"))
    p.add_argument("--artifacts", type=Path, default=Path("/data/muxiangyu/pythonPrograms/M2HImage/artifacts/m2h_ab_objective_v1"))
    p.add_argument("--out", type=Path, default=Path("/data/muxiangyu/pythonPrograms/M2HImage/artifacts/m2h_minimal_revalidation_20260907/protocol_v1"))
    args = p.parse_args()
    if args.out.exists():
        p.error("--out must not exist; protocol freeze outputs are write-once")
    for key in ("CUDA_VISIBLE_DEVICES",):
        os.environ[key] = ""
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "2"
    start = time.monotonic()
    pair_path = args.old_protocol / "dev128_pairs.json"
    full_dup = args.artifacts / "full_duplicates"
    group_dir = args.artifacts / "protocol_group_audit_v1/results"
    near_dir = args.artifacts / "protocol_near_duplicate_v1/results"
    required = [
        pair_path,
        full_dup / "per_image.jsonl",
        full_dup / "exact_groups.json",
        group_dir / "candidate_all.csv",
        near_dir / "image_pairs.csv.gz",
        Path(__file__).resolve(),
    ]
    for path in required:
        require(path.is_file(), f"missing input file: {path}")
    pairs_payload = load_pairs(pair_path)
    images, per_image = load_per_image(full_dup / "per_image.jsonl")
    candidate_rows = load_csv(group_dir / "candidate_all.csv")
    meta = {r["id"]: r for r in candidate_rows}
    require(len(meta) == len(candidate_rows) and meta, "invalid candidate_all.csv IDs")
    for (sid, role), row in per_image.items():
        if sid in meta:
            expected_path = meta[sid][role + "_path"]
            require(row["path"] == expected_path, f"per_image/candidate path mismatch: {(sid, role)}")
    dev_ids = {p["mid"] for p in pairs_payload["pairs"]} | {p["jid"] for p in pairs_payload["pairs"]}
    missing_ids = sorted(sid for sid in dev_ids if sid not in meta or (sid, "human") not in per_image)
    require(not missing_ids, f"frozen pair IDs missing from audits: {missing_ids[:5]}")
    train_humans_all = [r for r in images if r["split"] == "train" and r["role"] == "human"]
    train_sha = {r["sha256"] for r in train_humans_all}
    exact_overlaps = exact_human_train_overlaps(full_dup / "exact_groups.json", per_image)
    dev_exact_hits = []
    for sid in sorted(dev_ids):
        row = per_image[(sid, "human")]
        if row["sha256"] in train_sha:
            dev_exact_hits.append({"id": sid, "sha256": row["sha256"], "path": row["path"]})
    require(not dev_exact_hits, f"dev128 target/reference human exact train hits: {dev_exact_hits[:5]}")
    train_overlap_sha = {r["sha256"] for r in exact_overlaps}
    train_only_humans = [r for r in train_humans_all if r["sha256"] not in train_overlap_sha]
    calibration = calibrate(train_only_humans, 32)
    near_rows = collect_near_candidates(near_dir / "image_pairs.csv.gz", images, dev_ids)
    scored_rows = []
    for row in near_rows:
        score = metrics(row["dev_path"], row["train_path"])
        record = {**row, **score}
        record["high_risk"] = str(high_risk(record, calibration["threshold"])).lower()
        scored_rows.append(record)
    risky_ids = {r["dev_id"] for r in scored_rows if r["high_risk"] == "true"}
    full_pairs = []
    sensitivity_pairs = []
    ref_cluster_counts = Counter()
    m_group_counts = Counter()
    for i, row in enumerate(pairs_payload["pairs"]):
        m_meta, j_meta = meta[row["mid"]], meta[row["jid"]]
        enriched = {
            "pair_index": i,
            **row,
            "m_source_group": row.get("m_source_group", m_meta.get("source_group", "")),
            "j_source_group": row.get("j_source_group", j_meta.get("source_group", "")),
            "m_component_id": m_meta.get("component_id", ""),
            "j_component_id": j_meta.get("component_id", ""),
            "m_human_sha256": per_image[(row["mid"], "human")]["sha256"],
            "j_human_sha256": per_image[(row["jid"], "human")]["sha256"],
            "near_duplicate_sensitivity_remove": row["mid"] in risky_ids or row["jid"] in risky_ids,
        }
        ref_cluster_counts[enriched["j_component_id"]] += 1
        m_group_counts[enriched["m_source_group"]] += 1
        full_pairs.append(enriched)
        if not enriched["near_duplicate_sensitivity_remove"]:
            sensitivity_pairs.append(enriched)
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "frozen_dev128_pairs_unchanged.json", pairs_payload)
    write_csv(out / "exact_human_train_overlaps.csv",
              ["group_index", "sha256", "id", "split", "train_ids", "path"], exact_overlaps)
    write_json(out / "near_duplicate_threshold_calibration.json", calibration)
    near_fields = ["dev_id", "dev_role", "dev_path", "train_id", "train_path", "dhash_hamming", "sha256_equal",
                   "ssim_384x512", "center_ssim", "mae_384x512", "center_mae", "high_risk"]
    write_csv(out / "near_duplicate_candidates_scored.csv", near_fields, scored_rows)
    write_csv(out / "near_duplicate_high_risk.csv", near_fields, (r for r in scored_rows if r["high_risk"] == "true"))
    manifest = {
        "schema": SCHEMA,
        "purpose": "A4/B2-cont dev128 protocol revalidation freeze; no model, feature, generation, evaluation, split, or pair mutation",
        "pairs_preserved_from": str(pair_path),
        "pairs_preserved_sha256": sha256(pair_path),
        "pair_count": len(pairs_payload["pairs"]),
        "m_count": len({p["mid"] for p in pairs_payload["pairs"]}),
        "reference_count": len({p["jid"] for p in pairs_payload["pairs"]}),
        "pairs": full_pairs,
        "full_pair_indices": list(range(len(full_pairs))),
        "sensitivity_pair_indices": [p["pair_index"] for p in sensitivity_pairs],
        "near_duplicate_sensitivity_removed_pair_indices": [p["pair_index"] for p in full_pairs if p["near_duplicate_sensitivity_remove"]],
        "counts": {
            "exact_human_train_overlap_records_original_split": len(exact_overlaps),
            "dev128_exact_train_human_hits": len(dev_exact_hits),
            "near_duplicate_candidates": len(scored_rows),
            "near_duplicate_high_risk_dev_ids": len(risky_ids),
            "full_pairs": len(full_pairs),
            "sensitivity_pairs": len(sensitivity_pairs),
            "removed_pairs": len(full_pairs) - len(sensitivity_pairs),
        },
        "m_source_group_counts": dict(sorted(m_group_counts.items())),
        "reference_file_cluster_counts": dict(sorted(ref_cluster_counts.items())),
        "near_duplicate_threshold": calibration["threshold"],
        "threshold_qualified": calibration["threshold"]["qualified"],
        "status": "threshold_not_qualified_scores_retained" if not calibration["threshold"]["qualified"] else "threshold_qualified_for_sensitivity",
        "formal_split_ready": False,
        "main_revalidation_blocked": False,
        "limitations": calibration["limitations"] + [
            "Near-duplicate candidates are restricted to protocol_near_duplicate_v1 dHash-indexed image pairs.",
            "High-risk filtering is a fixed sensitivity subset only; original splits and frozen pairs are not edited.",
        ],
    }
    write_json(out / "manifest.json", manifest)
    provenance = {str(path): sha256(path) for path in required}
    provenance[str(out / "frozen_dev128_pairs_unchanged.json")] = sha256(out / "frozen_dev128_pairs_unchanged.json")
    write_json(out / "provenance_sha256.json", provenance)
    summary = {
        "schema": SCHEMA,
        "ok": True,
        "out": str(out),
        "pair_count": manifest["pair_count"],
        "m_count": manifest["m_count"],
        "reference_count": manifest["reference_count"],
        "exact_human_train_overlap_records_original_split": len(exact_overlaps),
        "dev128_exact_train_human_hits": 0,
        "near_duplicate_candidates": len(scored_rows),
        "near_duplicate_high_risk_dev_ids": len(risky_ids),
        "sensitivity_pairs": len(sensitivity_pairs),
        "removed_pairs": len(full_pairs) - len(sensitivity_pairs),
        "threshold_qualified": calibration["threshold"]["qualified"],
        "cpu_only": True,
        "gpu_tasks_started": False,
        "seconds": time.monotonic() - start,
    }
    write_json(out / "summary.json", summary)
    write_json(out / "artifact_sha256.json", {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()})
    with (out / "PROTOCOL_REVALIDATED").open("x") as f:
        f.write(sha256(out / "manifest.json") + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
