#!/usr/bin/env python3
"""Compute masked background LPIPS for the existing background report."""
from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from skimage.metrics import structural_similarity


def hist_cosine(a, b):
    va = np.concatenate([np.histogram(a[:, i], bins=16, range=(0, 256))[0] for i in range(3)]).astype(float)
    vb = np.concatenate([np.histogram(b[:, i], bins=16, range=(0, 256))[0] for i in range(3)]).astype(float)
    den = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return float(np.dot(va, vb) / den) if den else float("nan")


def load_item(row, size=128):
    human = Path(row["human_path"])
    mannequin = Path(row["mannequin_path"])
    hm = Path(row["human_mask_path"])
    mm = Path(row["mannequin_mask_path"])
    with Image.open(human) as x:
        h = np.asarray(x.convert("RGB").resize((size, size), Image.Resampling.BILINEAR))
        hs = x.size
    with Image.open(mannequin) as x:
        m = np.asarray(x.convert("RGB").resize((size, size), Image.Resampling.BILINEAR))
        ms = x.size
    with Image.open(hm) as x:
        hmask = np.asarray(x.convert("L").resize((size, size), Image.Resampling.NEAREST)) > 0
        hms = x.size
    with Image.open(mm) as x:
        mmask = np.asarray(x.convert("L").resize((size, size), Image.Resampling.NEAREST)) > 0
        mms = x.size
    if hs != ms or hms != hs or mms != ms:
        raise ValueError(f"size_mismatch:human={hs},mannequin={ms},human_mask={hms},mannequin_mask={mms}")
    bg = ~(hmask | mmask)
    ratio = float(bg.mean())
    if ratio < 0.01:
        raise ValueError(f"background_area_insufficient:{ratio:.6f}")
    hg = h.astype(np.float32).mean(axis=2) / 255.0
    mg = m.astype(np.float32).mean(axis=2) / 255.0
    ssim_map = structural_similarity(
        hg, mg, data_range=1.0, win_size=11, gaussian_weights=True,
        sigma=1.5, use_sample_covariance=False, full=True
    )[1]
    valid_ssim = binary_erosion(bg, structure=np.ones((11, 11), dtype=bool))
    if not valid_ssim.any():
        raise ValueError("background_area_insufficient_for_ssim_window")
    ssim_value = float(ssim_map[valid_ssim].mean())
    # Keep background pixels and replace foreground with each image's background mean.
    out = []
    for image in (h, m):
        mean = image[bg].mean(axis=0) if bg.any() else np.zeros(3)
        image = image.astype(np.float32) / 127.5 - 1.0
        fill = mean.astype(np.float32) / 127.5 - 1.0
        image[~bg] = fill
        out.append(image.transpose(2, 0, 1))
    return np.stack(out, axis=0).astype(np.float32), bg.astype(np.float32), ratio, ssim_value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--report-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--output-csv", default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    report = Path(args.report_dir)
    src = Path(args.output_csv) if args.output_csv else report / "background_consistency.csv"
    tmp = report / "background_consistency.lpips.partial.csv"
    rows = list(csv.DictReader(src.open(newline=""))) if src.is_file() else []
    # Recover the canonical row metadata if a prior interrupted writer left
    # an incomplete CSV. The person manifest is the authoritative split list.
    if not rows or not rows[0].get("id") or not rows[0].get("split"):
        pm = report / "person_disjoint_manifest.csv"
        formal = {"train", "val", "test"}
        rows = []
        for rec in csv.DictReader(pm.open(newline="")):
            split = rec.get("clean_split") or rec.get("split")
            if split not in formal or rec.get("action") == "quarantine":
                continue
            sid = rec["id"]
            hp = Path(rec["human_path"])
            mp = Path(rec["mannequin_path"])
            rows.append({
                "id": sid, "split": split, "human_path": str(hp),
                "mannequin_path": str(mp),
                "human_mask_path": str(hp.parents[2] / "human_parsing/fashn/masks/human" / f"{sid}.png"),
                "mannequin_mask_path": str(mp.parents[2] / "human_parsing/fashn/masks/mannequin" / f"{sid}.png"),
                "mask_source": "human_parsing/fashn",
                "status": "ok", "reason": "",
            })
    if args.num_shards > 1:
        rows = rows[args.shard_index::args.num_shards]
    if args.limit:
        rows = rows[:args.limit]
    base_rows = {r["id"]: dict(r) for r in rows}
    done = {}
    if args.resume and tmp.is_file():
        with tmp.open(newline="") as f:
            done = {r["id"]: r for r in csv.DictReader(f)}
    import torch
    import lpips
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.set_grad_enabled(False)
    model = lpips.LPIPS(net="alex", spatial=True).eval().to(device)
    model_name = "LPIPS-Alex v0.1 spatial masked background"
    fields = list(rows[0].keys()) + [
        "width", "height", "background_pixel_ratio", "foreground_pixel_ratio",
        "histogram_cosine", "ssim", "lpips_device", "lpips_status", "lpips"
    ]
    fields = list(dict.fromkeys(fields))
    results = dict(done)
    ssim_updates = {}
    batch_ids, batch_tensors, batch_masks = [], [], []

    def flush():
        nonlocal batch_ids, batch_tensors, batch_masks
        if not batch_ids:
            return
        x = torch.from_numpy(np.stack(batch_tensors)).to(device)
        a, b = x[:, 0], x[:, 1]
        # LPIPS spatial output is upsampled to input size for this model.
        dist = model(a, b).squeeze(1).detach().cpu().numpy()
        masks = np.stack(batch_masks)
        vals = []
        for i, sid in enumerate(batch_ids):
            mask = masks[i]
            denom = float(mask.sum())
            vals.append(float((dist[i] * mask).sum() / denom) if denom else None)
        for sid, val in zip(batch_ids, vals):
            r = dict(base_rows[sid])
            r.update(results.get(sid, {}))
            r["lpips_device"] = str(device)
            r["lpips_status"] = "ok" if val is not None else "invalid_background_measurement"
            r["lpips"] = "" if val is None else f"{val:.10f}"
            if sid in ssim_updates:
                r["ssim"] = f"{ssim_updates[sid]:.10f}"
            results[sid] = r
        batch_ids, batch_tensors, batch_masks = [], [], []
        with tmp.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(results[sid] for sid in [r["id"] for r in rows] if sid in results)

    for idx, row in enumerate(rows):
        sid = row["id"]
        if sid in done and done[sid].get("lpips_status") == "ok":
            continue
        try:
            pair, bg, _, ssim_value = load_item(row)
            raw_h = np.asarray(Image.open(row["human_path"]).convert("RGB"))
            raw_m = np.asarray(Image.open(row["mannequin_path"]).convert("RGB"))
            if raw_h.shape != raw_m.shape:
                raise ValueError("image_size_mismatch")
            row.update({
                "width": raw_h.shape[1], "height": raw_h.shape[0],
                "background_pixel_ratio": float(bg.mean()),
                "foreground_pixel_ratio": float(1.0 - bg.mean()),
                "histogram_cosine": hist_cosine(raw_h.reshape(-1, 3), raw_m.reshape(-1, 3)),
            })
            ssim_updates[sid] = ssim_value
            results[sid] = dict(row)
            batch_ids.append(sid)
            batch_tensors.append(pair)
            batch_masks.append(bg)
        except Exception as exc:
            r = dict(row)
            r["lpips_device"] = str(device)
            r["lpips_status"] = "invalid_background_measurement"
            r["lpips"] = ""
            r["reason"] = (r.get("reason", "") + ";" if r.get("reason") else "") + f"lpips:{type(exc).__name__}:{exc}"
            results[sid] = r
        if len(batch_ids) >= args.batch_size:
            flush()
        if (idx + 1) % 1000 == 0:
            print(f"processed={idx+1}/{len(rows)} completed={len(results)}", flush=True)
    flush()
    out_rows = []
    for r in rows:
        item = results.get(r["id"], dict(r, lpips_device=str(device), lpips_status="missing", lpips=""))
        if r["id"] in ssim_updates:
            item["ssim"] = f"{ssim_updates[r['id']]:.10f}"
        out_rows.append(item)
    with src.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    if tmp.exists():
        tmp.unlink()
    vals = [float(r["lpips"]) for r in out_rows if r.get("lpips_status") == "ok" and r.get("lpips") not in ("", None)]
    by_split = {}
    for split in ("train", "val", "test"):
        sv = [float(r["lpips"]) for r in out_rows if r["split"] == split and r.get("lpips_status") == "ok" and r.get("lpips") not in ("", None)]
        by_split[split] = {"count": len(sv), "mean": float(np.mean(sv)) if sv else None, "median": float(np.median(sv)) if sv else None}
    summary = json.loads((report / "background_consistency_summary.json").read_text()) if (report / "background_consistency_summary.json").is_file() else {}
    summary.update({
        "lpips_available": True,
        "lpips_count": len(vals),
        "lpips_invalid_count": len(out_rows) - len(vals),
        "lpips_mean": float(np.mean(vals)) if vals else None,
        "lpips_median": float(np.median(vals)) if vals else None,
        "lpips_p05": float(np.quantile(vals, .05)) if vals else None,
        "lpips_p95": float(np.quantile(vals, .95)) if vals else None,
        "lpips_by_split": by_split,
        "lpips_scope": "all formal person-disjoint train/val/test samples",
        "lpips_model": model_name,
        "lpips_preprocess": "128x128 bilinear RGB, [-1,1], union foreground exclusion, spatial LPIPS weighted by common background",
        "lpips_device": str(device),
        "lpips_completed_at": datetime.now(timezone.utc).isoformat(),
    })
    if args.num_shards == 1:
        (report / "background_consistency_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"count": len(out_rows), "lpips_count": len(vals), "invalid": len(out_rows)-len(vals), "mean": summary["lpips_mean"], "by_split": by_split}, indent=2))


if __name__ == "__main__":
    main()
