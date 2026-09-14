#!/usr/bin/env python3
"""Measure human/mannequin background consistency for a clean split manifest.

The script deliberately keeps the metric implementation dependency-light:
Pillow is required, NumPy is used for efficient RGB histograms and SSIM, and
LPIPS is optional. Missing optional dependencies are recorded as invalid
measurements instead of being silently replaced.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

try:
    import numpy as np
    NUMPY_ERROR = ""
except ImportError as exc:  # pragma: no cover - exercised on minimal hosts
    np = None
    NUMPY_ERROR = str(exc)
try:
    from PIL import Image
    PIL_ERROR = ""
except ImportError as exc:  # pragma: no cover - exercised on minimal hosts
    Image = None
    PIL_ERROR = str(exc)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def manifest_files(dataset_root: Path, split_manifest: Path) -> list[tuple[str, Path]]:
    files = sorted(split_manifest.glob("*.txt")) if split_manifest.is_dir() else [split_manifest]
    out: list[tuple[str, Path]] = []
    for path in files:
        split = path.stem.split("_", 1)[0]
        if split not in {"train", "val", "test"}:
            continue
        for line in path.read_text().splitlines():
            sample_id = line.strip()
            if sample_id:
                out.append((split, Path(sample_id)))
    return out


def resolve_mask(dataset_root: Path, role: str, sample_id: str) -> Path:
    roots = [dataset_root]
    sibling = dataset_root.parent / "M2H_Final_v2"
    if sibling != dataset_root:
        roots.append(sibling)
    candidates = [
        base / rel / role / f"{sample_id}.png"
        for base in roots
        for rel in ("human_parsing/fashn/masks", "clothes_bySAM/masks", "derived/region_masks")
    ]
    candidates += [base / "derived/region_masks" / f"{sample_id}.png" for base in roots]
    return next((p for p in candidates if p.is_file()), candidates[0])


def resolve_image(path: Path, dataset_root: Path, role: str, sample_id: str) -> Path:
    if path.is_file():
        return path
    sibling = dataset_root.parent / "M2H_Final_v2"
    candidates = [sibling / "images" / role / path.name,
                  sibling / "images" / role / f"{sample_id}.jpg",
                  sibling / "images" / role / f"{sample_id}.png"]
    return next((p for p in candidates if p.is_file()), path)


def load_rgb(path: Path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB")), im.size


def load_rgb_pillow(path: Path):
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        return rgb.copy(), rgb.size


def load_mask(path: Path, size: tuple[int, int]):
    with Image.open(path) as im:
        arr = np.asarray(im.convert("L"))
        original_size = im.size
    if original_size != size:
        return None, original_size
    return arr > 0, original_size


def hist_cosine(a, b) -> float:
    # 16 bins/channel, normalized over valid background pixels.
    va = np.concatenate([np.histogram(a[:, i], bins=16, range=(0, 256))[0] for i in range(3)]).astype(float)
    vb = np.concatenate([np.histogram(b[:, i], bins=16, range=(0, 256))[0] for i in range(3)]).astype(float)
    den = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return float(np.dot(va, vb) / den) if den else float("nan")


def ssim_gray(a, b) -> float:
    af = a.astype(np.float64) / 255.0
    bf = b.astype(np.float64) / 255.0
    mu_a, mu_b = af.mean(), bf.mean()
    va, vb = af.var(), bf.var()
    cov = ((af - mu_a) * (bf - mu_b)).mean()
    c1, c2 = 0.01**2, 0.03**2
    den = (mu_a * mu_a + mu_b * mu_b + c1) * (va + vb + c2)
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / den) if den else float("nan")


def pillow_metrics(human, mannequin, hmask, mmask):
    """Downsampled pure-Pillow metrics for hosts without NumPy."""
    size = (128, 128)
    ha = human.resize(size, Image.Resampling.BILINEAR)
    ma = mannequin.resize(size, Image.Resampling.BILINEAR)
    hm = hmask.resize(size, Image.Resampling.NEAREST)
    mm = mmask.resize(size, Image.Resampling.NEAREST)
    hp, mp = list(ha.getdata()), list(ma.getdata())
    hmp, mmp = list(hm.getdata()), list(mm.getdata())
    bg = [i for i, (x, y) in enumerate(zip(hmp, mmp)) if not (x or y)]
    if len(bg) < 128:
        raise ValueError("background_area_insufficient")
    bins_a = [0] * 48
    bins_b = [0] * 48
    ga, gb = [], []
    for i in bg:
        pa, pb = hp[i], mp[i]
        for c in range(3):
            bins_a[c * 16 + min(pa[c] // 16, 15)] += 1
            bins_b[c * 16 + min(pb[c] // 16, 15)] += 1
        ga.append(sum(pa) / 3.0)
        gb.append(sum(pb) / 3.0)
    den = math.sqrt(sum(x * x for x in bins_a) * sum(x * x for x in bins_b))
    cosine = sum(x * y for x, y in zip(bins_a, bins_b)) / den if den else float("nan")
    ma_v, mb_v = sum(ga) / len(ga), sum(gb) / len(gb)
    va = sum((x - ma_v) ** 2 for x in ga) / len(ga)
    vb = sum((x - mb_v) ** 2 for x in gb) / len(gb)
    cov = sum((x - ma_v) * (y - mb_v) for x, y in zip(ga, gb)) / len(ga)
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2 * ma_v * mb_v / (255**2) + c1) *
            (2 * cov / (255**2) + c2)) / (
                (ma_v**2 / (255**2) + mb_v**2 / (255**2) + c1) *
                (va / (255**2) + vb / (255**2) + c2))
    return cosine, float(ssim), len(bg) / len(hp), (len(hmp) - len(bg)) / len(hp)


_LPIPS_MODEL = None
_LPIPS_DEVICE = None


def lpips_metric(a, b) -> tuple[float | None, str]:
    global _LPIPS_MODEL, _LPIPS_DEVICE
    try:
        import torch
        import lpips
    except ImportError as exc:
        return None, f"lpips_unavailable:{exc}"
    try:
        if _LPIPS_MODEL is None:
            _LPIPS_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _LPIPS_MODEL = lpips.LPIPS(net="alex").eval().to(_LPIPS_DEVICE)
        a = np.asarray(Image.fromarray(a).resize((128, 128), Image.Resampling.BILINEAR))
        b = np.asarray(Image.fromarray(b).resize((128, 128), Image.Resampling.BILINEAR))
        ta = torch.from_numpy(a.transpose(2, 0, 1)).float().div(127.5).sub(1).unsqueeze(0).to(_LPIPS_DEVICE)
        tb = torch.from_numpy(b.transpose(2, 0, 1)).float().div(127.5).sub(1).unsqueeze(0).to(_LPIPS_DEVICE)
        with torch.no_grad():
            return float(_LPIPS_MODEL(ta, tb).item()), "ok"
    except Exception as exc:
        return None, f"lpips_error:{type(exc).__name__}:{exc}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="/data/muxiangyu/datasets/M2HImage/M2H_Final_v2_clean_v1")
    ap.add_argument("--experiment-root", default="/data/muxiangyu/experiments/M2H_Final_v2_clean_v1")
    ap.add_argument("--split-manifest", default=None)
    ap.add_argument("--report-dir", default=None)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    root = Path(args.dataset_root)
    split_root = Path(args.split_manifest) if args.split_manifest else root / "splits_person_disjoint"
    report = Path(args.report_dir) if args.report_dir else root / "dataset_cleaning_report"
    report.mkdir(parents=True, exist_ok=True)
    rows = []
    if Image is None:
        reason = f"dependency_unavailable:Pillow:{PIL_ERROR}"
        for split, sid_path in manifest_files(root, split_root):
            rows.append({"id": str(sid_path), "split": split, "status": "invalid_background_measurement",
                         "reason": reason})
    else:
        # The clean manifest contains absolute source paths, so read it when available.
        manifest_path = report / "person_disjoint_manifest.csv"
        by_id = {}
        if manifest_path.is_file():
            with manifest_path.open(newline="") as f:
                by_id = {r["id"]: r for r in csv.DictReader(f)}
        for split, sid_path in manifest_files(root, split_root):
            sid = str(sid_path)
            rec = by_id.get(sid, {})
            human = resolve_image(Path(rec.get("human_path", root / f"images/human/{sid}.jpg")), root, "human", sid)
            mannequin = resolve_image(Path(rec.get("mannequin_path", root / f"images/mannequin/{sid}.png")), root, "mannequin", sid)
            hm = resolve_mask(root, "human", sid)
            mm = resolve_mask(root, "mannequin", sid)
            row = {"id": sid, "split": split, "human_path": str(human), "mannequin_path": str(mannequin),
                   "human_mask_path": str(hm), "mannequin_mask_path": str(mm),
                   "status": "ok", "reason": "", "histogram_cosine": "", "ssim": "", "lpips": ""}
            try:
                if not human.is_file() or not mannequin.is_file() or not hm.is_file() or not mm.is_file():
                    raise FileNotFoundError("missing human/mannequin image or foreground mask")
                if np is None:
                    hi, hs = load_rgb_pillow(human)
                    mi, ms = load_rgb_pillow(mannequin)
                    with Image.open(hm) as x:
                        hmi = x.convert("L").copy()
                    with Image.open(mm) as x:
                        mmi = x.convert("L").copy()
                    if hs != ms or hmi.size != hs or mmi.size != ms:
                        raise ValueError(f"image_or_mask_size_mismatch:{hs}:{ms}:{hmi.size}:{mmi.size}")
                    cosine, ssim, valid_ratio, fg_ratio = pillow_metrics(hi, mi, hmi, mmi)
                else:
                    ha, hs = load_rgb(human)
                    ma, ms = load_rgb(mannequin)
                    if hs != ms:
                        raise ValueError(f"image_size_mismatch:{hs}!={ms}")
                    hmask, hms = load_mask(hm, hs)
                    mmask, mms = load_mask(mm, ms)
                    if hmask is None or mmask is None:
                        raise ValueError(f"mask_size_mismatch:{hms}!={mms}")
                    bg = ~(hmask | mmask)
                    valid_ratio = float(bg.mean())
                    fg_ratio = float((hmask | mmask).mean())
                row.update({"width": hs[0], "height": hs[1], "background_pixel_ratio": valid_ratio,
                            "foreground_pixel_ratio": fg_ratio, "mask_source": "human_parsing/fashn"})
                if valid_ratio < 0.01 or fg_ratio > 0.99:
                    raise ValueError("background_area_insufficient")
                if np is not None:
                    a, b = ha[bg], ma[bg]
                    row["histogram_cosine"] = hist_cosine(a, b)
                    row["ssim"] = ssim_gray(ha.mean(axis=2)[bg], ma.mean(axis=2)[bg])
                    masked_a, masked_b = ha.copy(), ma.copy()
                    masked_a[~bg] = 0
                    masked_b[~bg] = 0
                    lp, lp_status = lpips_metric(masked_a, masked_b)
                else:
                    row["histogram_cosine"], row["ssim"] = cosine, ssim
                    lp, lp_status = None, "lpips_unavailable:numpy is required"
                row["lpips"] = "" if lp is None else lp
                if lp_status != "ok":
                    row["reason"] = lp_status
            except Exception as exc:
                row["status"] = "invalid_background_measurement"
                row["reason"] = f"{type(exc).__name__}:{exc}"
            rows.append(row)
    fields = sorted({k for r in rows for k in r})
    with (report / "background_consistency.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    valid = [r for r in rows if r.get("status") == "ok" and r.get("histogram_cosine") not in ("", None)]
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root), "split_manifest": str(split_root),
        "sample_count": len(rows), "valid_count": len(valid),
        "invalid_count": len(rows) - len(valid),
        "status_counts": {s: sum(r.get("status") == s for r in rows) for s in ["ok", "invalid_background_measurement"]},
        "split_counts": {s: sum(r.get("split") == s for r in rows) for s in ["train", "val", "test"]},
        "lpips_available": any(r.get("lpips") not in ("", None) for r in rows),
        "input_sha256": sha256(manifest_path) if (manifest_path := report / "person_disjoint_manifest.csv").is_file() else None,
    }
    if valid:
        for key in ("histogram_cosine", "ssim", "lpips"):
            vals = [float(r[key]) for r in valid if r.get(key) not in ("", None)]
            summary[f"{key}_mean"] = sum(vals) / len(vals) if vals else None
    (report / "background_consistency_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 2 if args.strict and summary["invalid_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
