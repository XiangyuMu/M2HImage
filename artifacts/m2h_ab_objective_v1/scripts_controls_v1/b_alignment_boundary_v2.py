"""CPU-only B controls, fixed input-M scoring regions, synthetic seam calibration.

No H, face recognizer, learned model, downloaded dependency, or training is used.
Arrays are RGB uint8. All displacements/radii are in native-resolution pixels.
ECC returns a destination-to-source sampling displacement: output(x)=M(x+d).
"""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np


METHODS = ("base", "inplace_feather", "warp_feather", "warp_multiband")
PROXIES = ("boundary_cross_rgb_mean", "boundary_cross_rgb_p95",
           "boundary_cross_excess_mean", "boundary_gradient_p95",
           "boundary_laplacian_p95", "transition_gradient_p95", "transition_laplacian_p95")


@dataclass(frozen=True)
class Config:
    max_shift_px: float = 6.0
    fb_limit_px: float = 1.0
    min_ecc: float = 0.5
    min_gray_std: float = 0.015
    min_valid_fraction: float = 0.90
    ecc_max_side: int = 512
    ecc_iterations: int = 60
    ecc_epsilon: float = 1e-5
    erosion_px: int = 4
    feather_px: int = 8
    boundary_radius_px: int = 8
    pyramid_levels: int = 4
    texture_quantile: float = 0.75
    texture_min_gradient: float = 0.02
    spill_threshold_u8: int = 2


def fixed_regions(mask, radius=8):
    """Input-only partition. Never takes output masks, displacement, or q."""
    mask = np.asarray(mask, dtype=bool)
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    inner = cv2.erode(mask.astype(np.uint8), kernel,
                      borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    outer = cv2.dilate(mask.astype(np.uint8), kernel,
                       borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    distance = cv2.distanceTransform(np.pad(mask.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    return {"garment": mask, "interior": inner, "boundary": outer & ~inner,
            "transition": mask & (distance <= 16),
            "exterior": ~outer}


def sample_translation(image, displacement, nearest=False):
    """Backward warp: result[y,x] = image[y+dy,x+dx]; no sign inference.

    Validity excludes border extrapolation, even though reflected pixels are
    provided for filtering. It must gate editing, never scoring.
    """
    dx, dy = map(float, displacement)
    if not np.isfinite([dx, dy]).all():
        raise ValueError("nonfinite displacement")
    h, w = image.shape[:2]
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    x, y = x + dx, y + dy
    valid = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    result = cv2.remap(image, x, y, cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT_101)
    return result, valid


def correspondence_support(mask, displacement):
    # Linear interpolation needs all contributing source pixels inside M.
    sampled, bounds = sample_translation(mask.astype(np.float32), displacement)
    return bounds & (sampled >= 1 - 1e-6) & mask.astype(bool)


def alignment_decision(forward, backward, correlations, mask, config=Config()):
    """Pure acceptance policy; rejected alignments explicitly use identity."""
    forward, backward = np.asarray(forward, float), np.asarray(backward, float)
    finite = np.isfinite(np.r_[forward, backward, correlations]).all()
    reasons = []
    fb = float(np.linalg.norm(forward + backward)) if finite else None
    displacement = float(np.linalg.norm(forward)) if finite else None
    reverse_displacement = float(np.linalg.norm(backward)) if finite else None
    coverage = float(correspondence_support(mask, forward)[mask.astype(bool)].mean()) if finite and mask.any() else 0.0
    if not finite:
        reasons.append("nonfinite_ecc")
    else:
        if max(displacement, reverse_displacement) > config.max_shift_px:
            reasons.append("displacement_cap")
        if fb > config.fb_limit_px:
            reasons.append("forward_backward_inconsistent")
        if min(correlations) < config.min_ecc:
            reasons.append("low_ecc_correlation")
    if coverage < config.min_valid_fraction:
        reasons.append("low_source_support_coverage")
    accepted = not reasons
    return {"accepted": accepted, "status": "accepted" if accepted else "identity_fallback",
            "reasons": reasons, "candidate_sample_dxdy": forward.tolist() if finite else None,
            "reverse_sample_dxdy": backward.tolist() if finite else None,
            "ecc_forward": float(correlations[0]) if finite else None,
            "ecc_backward": float(correlations[1]) if finite else None,
            "candidate_shift_norm_px": displacement, "reverse_shift_norm_px": reverse_displacement,
            "fb_error_px": fb, "candidate_source_support_fraction_of_M": coverage,
            "applied_sample_dxdy": forward.tolist() if accepted else [0.0, 0.0]}


def estimate_alignment(source, base, mask, config=Config()):
    """Two zero-initialized, CPU ECC translations; no coarse/global search.

    findTransformECC(template=base,input=source) estimates the sampling map
    from base coordinates into source (use WARP_INVERSE_MAP if warpAffine).
    The reversed call must return its inverse. This is a global consistency
    proxy, not an occlusion/visibility annotation or local flow guarantee.
    """
    start = time.monotonic()
    try:
        h, w = mask.shape
        scale = min(1.0, config.ecc_max_side / max(h, w))
        size = (max(8, round(w * scale)), max(8, round(h * scale)))
        sx, sy = size[0] / w, size[1] / h
        gray = [cv2.resize(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255,
                           size, interpolation=cv2.INTER_AREA) for im in (source, base)]
        # Conservative source-only eroded ROI; same fixed coordinate ROI in both calls.
        safe = fixed_regions(mask, int(np.ceil(config.max_shift_px)) + 2)["interior"]
        roi = cv2.resize(safe.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
        if int(roi.sum()) < 64:
            raise ValueError("insufficient_eroded_input_roi")
        if min(float(im[roi.astype(bool)].std()) for im in gray) < config.min_gray_std:
            raise ValueError("insufficient_gray_texture")
        criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
                    config.ecc_iterations, config.ecc_epsilon)
        shifts, correlations = [], []
        for template, image in ((gray[1], gray[0]), (gray[0], gray[1])):
            corr, warp = cv2.findTransformECC(template, image, np.eye(2, 3, dtype=np.float32),
                                             cv2.MOTION_TRANSLATION, criteria, roi * 255, 5)
            shifts.append([float(warp[0, 2]) / sx, float(warp[1, 2]) / sy])
            correlations.append(float(corr))
        result = alignment_decision(*shifts, correlations, mask, config)
        result.update(ecc_size=list(size), ecc_roi_pixels=int(roi.sum()))
    except (cv2.error, ValueError) as exc:
        result = {"accepted": False, "status": "identity_fallback", "reasons": [str(exc)],
                  "applied_sample_dxdy": [0.0, 0.0], "candidate_sample_dxdy": None,
                  "candidate_source_support_fraction_of_M": None, "fb_error_px": None}
    result["seconds"] = time.monotonic() - start
    return result


def feather_alpha(mask, config=Config()):
    # Zero padding gives defined distances even when M touches the image frame.
    dist = cv2.distanceTransform(np.pad(mask.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    return np.clip((dist - config.erosion_px) / config.feather_px, 0, 1).astype(np.float32)


def feather_blend(source, base, alpha):
    out = alpha[..., None] * source.astype(np.float32) + (1 - alpha[..., None]) * base
    return np.rint(out).clip(0, 255).astype(np.uint8)


def multiband_blend(source, base, alpha, levels=4):
    """Gaussian alpha / Laplacian RGB pyramids; exact support restored last."""
    if np.array_equal(source, base) or not np.any(alpha > 0):
        return base.copy()
    gs, gb, ga = [source.astype(np.float32)], [base.astype(np.float32)], [alpha.astype(np.float32)]
    for _ in range(levels - 1):
        if min(gs[-1].shape[:2]) < 4:
            break
        gs.append(cv2.pyrDown(gs[-1]))
        gb.append(cv2.pyrDown(gb[-1]))
        ga.append(cv2.pyrDown(ga[-1]))
    out = ga[-1][..., None] * gs[-1] + (1 - ga[-1][..., None]) * gb[-1]
    for i in range(len(gs) - 2, -1, -1):
        size = (gs[i].shape[1], gs[i].shape[0])
        ls = gs[i] - cv2.pyrUp(gs[i + 1], dstsize=size)
        lb = gb[i] - cv2.pyrUp(gb[i + 1], dstsize=size)
        out = cv2.pyrUp(out, dstsize=size) + ga[i][..., None] * ls + (1 - ga[i][..., None]) * lb
    out = np.rint(out).clip(0, 255).astype(np.uint8)
    out[alpha <= 0] = base[alpha <= 0]
    return out


def method_output(method, source, base, mask, alignment, config=Config()):
    alpha = feather_alpha(mask, config)
    support = mask.astype(bool)
    donor = source
    if method == "base":
        alpha = np.zeros_like(alpha)
    elif method.startswith("warp_"):
        donor, _ = sample_translation(source, alignment["applied_sample_dxdy"])
        support = correspondence_support(mask, alignment["applied_sample_dxdy"])
        alpha = alpha * support
    elif method != "inplace_feather":
        raise ValueError("unknown method: " + method)
    if method == "base":
        out = base.copy()
    elif method == "warp_multiband":
        out = multiband_blend(donor, base, alpha, config.pyramid_levels)
    else:
        out = feather_blend(donor, base, alpha)
    m = mask.astype(bool)
    coverage = {"source_support_fraction_of_M": float(support[m].mean()),
                "alpha_positive_fraction_of_M": float((alpha[m] > 0).mean()),
                "alpha_mean_on_M": float(alpha[m].mean()),
                "alpha_p10_on_M": float(np.quantile(alpha[m], .1)),
                "alpha_p90_on_M": float(np.quantile(alpha[m], .9)),
                "changed_fraction_of_M": float(np.any(out != base, axis=2)[m].mean())}
    return out, alpha, coverage


def image_features(image):
    f = image.astype(np.float32) / 255
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8)
    return {"rgb": f, "gx": gx, "gy": gy,
            "gradient": np.sqrt(np.mean(gx * gx + gy * gy, axis=2)),
            "hf": f - cv2.GaussianBlur(f, (0, 0), 2),
            "laplacian": np.mean(np.abs(cv2.Laplacian(f, cv2.CV_32F, ksize=1)), axis=2)}


def reference_context(source, mask, config=Config()):
    features = image_features(source)
    regions = fixed_regions(mask, config.boundary_radius_px)
    values = features["gradient"][regions["garment"]]
    threshold = max(config.texture_min_gradient, float(np.quantile(values, config.texture_quantile))) if len(values) else config.texture_min_gradient
    regions["strong_texture"] = regions["garment"] & (features["gradient"] >= threshold)
    return {"features": features, "regions": regions, "texture_threshold": threshold}


def crossing_statistics(image, mask):
    """RGB jumps at frozen M crossing edges, plus local jump excess.

    Excess = max(0, crossing jump - mean(adjacent same-side jumps)). A real
    garment edge/shadow can score high; a blur can score low. Not a defect label.
    """
    f = image.astype(np.float32) / 255
    jumps, excesses = [], []
    for axis in (0, 1):
        a, m = np.moveaxis(f, axis, 0), np.moveaxis(mask.astype(bool), axis, 0)
        d = np.sqrt(np.mean(np.diff(a, axis=0) ** 2, axis=2))
        crossing = m[:-1] != m[1:]
        jumps.append(d[crossing])
        if len(d) >= 3:
            eligible = crossing[1:-1] & ~crossing[:-2] & ~crossing[2:]
            excesses.append(np.maximum(0, d[1:-1] - .5 * (d[:-2] + d[2:]))[eligible])
    jumps = np.concatenate(jumps)
    excesses = np.concatenate(excesses) if excesses else np.array([])
    return {"boundary_cross_edges": int(jumps.size),
            "boundary_excess_edges": int(excesses.size),
            "boundary_cross_rgb_mean": float(jumps.mean()) if jumps.size else None,
            "boundary_cross_rgb_p95": float(np.quantile(jumps, .95)) if jumps.size else None,
            "boundary_cross_excess_mean": float(excesses.mean()) if excesses.size else None}


def measure(source, base, output, mask, config=Config(), context=None):
    """All denominators frozen from M. Source fidelity is not paired-H truth."""
    context = context or reference_context(source, mask, config)
    sf, of = context["features"], image_features(output)
    regions = context["regions"]
    squared = np.mean((sf["rgb"] - of["rgb"]) ** 2, axis=2)
    gradient_error = np.mean(np.abs(sf["gx"] - of["gx"]) + np.abs(sf["gy"] - of["gy"]), axis=2)
    hf_error = np.mean(np.abs(sf["hf"] - of["hf"]), axis=2)
    result = {"source_texture_gradient_threshold": context["texture_threshold"]}
    for name in ("garment", "interior", "boundary", "transition", "strong_texture"):
        roi = regions[name]
        result[name + "_area"] = int(roi.sum())
        for metric, values in (("mse_to_source", squared), ("gradient_l1_to_source", gradient_error),
                               ("hf_l1_to_source", hf_error)):
            result[name + "_" + metric] = float(values[roi].mean()) if roi.any() else None
    result.update(crossing_statistics(output, mask))
    band = regions["boundary"]
    for name in ("gradient", "laplacian"):
        result["boundary_" + name + "_p95"] = float(np.quantile(of[name][band], .95)) if band.any() else None
        transition = regions['transition']
        result['transition_' + name + '_p95'] = float(np.quantile(of[name][transition], .95)) if transition.any() else None
    difference = np.max(np.abs(output.astype(np.int16) - base.astype(np.int16)), axis=2)
    changed = difference > config.spill_threshold_u8
    for name, roi in (("outside_M", ~regions["garment"]), ("outside_allowed", regions["exterior"])):
        result[name + "_area"] = int(roi.sum())
        result[name + "_changed_pixels"] = int(changed[roi].sum())
        result[name + "_edit_fraction"] = float(changed[roi].mean()) if roi.any() else None
        result[name + "_max_abs_u8"] = int(difference[roi].max()) if roi.any() else None
    return result


def synthetic_perturbation(clean, mask, kind, severity):
    """Known synthetic clean target; mask/support never changes across levels."""
    m = mask.astype(bool)
    out = clean.copy()
    if kind == "clean":
        return out
    if kind in ("splice_offset", "feather_offset"):
        # Fixed RGB offset in uint8 units. Saturation is part of the recorded fixture.
        donor = np.rint(clean.astype(np.float32) + severity).clip(0, 255).astype(np.uint8)
    elif kind in ("misalignment", "feather_misalignment"):
        # A visual rightward translation uses sample displacement (-severity, 0).
        donor, _ = sample_translation(clean, (-float(severity), 0))
    else:
        raise ValueError("unknown perturbation")
    if kind.startswith('feather_'):
        return feather_blend(donor, clean, feather_alpha(m))
    out[m] = donor[m]
    return out


def calibration_cases():
    return [("clean", 0), ("splice_offset", 8), ("splice_offset", 24),
            ("misalignment", 1), ("misalignment", 3), ("misalignment", 6),
            ("feather_offset", 8), ("feather_offset", 24),
            ("feather_misalignment", 1), ("feather_misalignment", 3), ("feather_misalignment", 6)]


def summarize_calibration(rows):
    report = {"scope": "source-M self-perturbations; no paired human GT or real defect labels",
              "unit": "unique mid (not repeated identities)", "cases": {}}
    fields = PROXIES + ("synthetic_boundary_rgb_mae_to_known_clean",
                        "synthetic_transition_gradient_l1_to_known_clean",
                        "synthetic_boundary_gradient_l1_to_known_clean")
    clean = {r["mid"]: r for r in rows if r["kind"] == "clean"}
    for kind, severity in calibration_cases():
        selected = [r for r in rows if r["kind"] == kind and r["severity"] == severity]
        entry = {"n": len(selected), "metrics": {}}
        for field in fields:
            values = [r[field] for r in selected if r.get(field) is not None]
            deltas = [r[field] - clean[r["mid"]][field] for r in selected
                      if r.get(field) is not None and clean.get(r["mid"], {}).get(field) is not None]
            entry["metrics"][field] = {"n": len(values), "mean": float(np.mean(values)) if values else None,
                                        "paired_delta_from_clean_mean": float(np.mean(deltas)) if deltas else None,
                                        "fraction_increased_vs_clean": float(np.mean(np.array(deltas) > 1e-6)) if deltas else None}
        report["cases"][f"{kind}_{severity}"] = entry
    report["severity_trend"] = {}
    for kind in ("splice_offset", "misalignment", "feather_offset", "feather_misalignment"):
        report["severity_trend"][kind] = {}
        for field in fields:
            trends = []
            for mid in clean:
                ordered = sorted([r for r in rows if r["mid"] == mid and r["kind"] == kind], key=lambda r: r["severity"])
                if len(ordered) >= 2 and all(r.get(field) is not None for r in ordered):
                    trends.append(bool(np.all(np.diff([r[field] for r in ordered]) >= -1e-6)))
            report["severity_trend"][kind][field] = {"n": len(trends), "nondecreasing_fraction": float(np.mean(trends)) if trends else None}
    report["interpretation"] = "Sensitivity diagnostics only; no fitted threshold, real defect rate, human calibration, or formal gate. Failure to increase must be reported."
    return report


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("unreadable image: " + str(path))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_rgb(path, image):
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise OSError("failed to save " + str(path))


def safe_token(value):
    text = str(value)
    if not text or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in text):
        raise ValueError("invalid pair filename token")
    return text


def pair_name(pair):
    return f'{safe_token(pair["mid"])}__id{safe_token(pair["jid"])}__seed{safe_token(pair["seed"])}.png'


def source_path(root, mid):
    # Probe/M contains NPZ features, not the original source RGB.
    for ext in (".png", ".jpg", ".jpeg"):
        path = root / "images/mannequin" / (safe_token(mid) + ext)
        if path.is_file():
            return path
    raise FileNotFoundError("missing source RGB for " + str(mid))


def aggregate(rows):
    result = {}
    for method in METHODS:
        selected = [r for r in rows if r["method"] == method]
        valid = [r for r in selected if r["status"] == "ok"]
        fields = sorted({key for r in valid for key in r["metrics"]})
        result[method] = {"expected": len(selected), "ok": len(valid), "failed": len(selected) - len(valid),
                          "identity_fallback": sum(r.get("alignment_status") == "identity_fallback" for r in valid),
                          "metrics": {}}
        for field in fields:
            values = [r["metrics"][field] for r in valid if r["metrics"].get(field) is not None]
            result[method]["metrics"][field] = {"n": len(values), "missing_including_failures": len(selected) - len(values),
                "mean": float(np.mean(values)) if values else None,
                "p50": float(np.median(values)) if values else None,
                "p95": float(np.quantile(values, .95)) if values else None}
        result[method]["coverage"] = {}
        for field in sorted({key for r in valid for key in r["coverage"]}):
            values = [r["coverage"][field] for r in valid]
            result[method]["coverage"][field] = {"n": len(values), "mean": float(np.mean(values))}
    # Exploratory paired deltas only; no falsely independent image CI.
    base_rows = {r["name"]: r for r in rows if r["method"] == "base" and r["status"] == "ok"}
    for method in METHODS[1:]:
        pairs = [r for r in rows if r["method"] == method and r["status"] == "ok" and r["name"] in base_rows]
        result[method]["paired_delta_vs_base"] = {}
        for field in result[method]["metrics"]:
            values = [r["metrics"][field] - base_rows[r["name"]]["metrics"][field] for r in pairs
                      if r["metrics"].get(field) is not None and base_rows[r["name"]]["metrics"].get(field) is not None]
            result[method]["paired_delta_vs_base"][field] = {"n": len(values), "mean": float(np.mean(values)) if values else None}
    return result


def write_json(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def emit_row(handle, value):
    handle.write(json.dumps(value, allow_nan=False) + "\n")
    handle.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="dataset root (reads images/mannequin only)")
    parser.add_argument("--probe", type=Path, required=True, help="audit.json + outputs/ + input masks/")
    parser.add_argument("--out", type=Path, required=True, help="new directory; must not exist")
    parser.add_argument("--limit", type=int, default=0, help="first N audit pairs; 0=all, including 128 pairs")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-seconds", type=float, default=1800, help="soft run limit, checked between pairs")
    parser.add_argument("--dev-identity-layout", action="store_true", help="write legacy-name hardlink adapter; see notes")
    args = parser.parse_args()
    if args.limit < 0 or args.threads < 1 or args.max_seconds <= 0:
        parser.error("invalid budget/limit/threads")
    config = Config()  # Fixed configuration; no per-image/after-results threshold fitting.
    cv2.setNumThreads(args.threads)
    cv2.ocl.setUseOpenCL(False)
    cv2.setRNGSeed(0)
    audit_path = args.probe / "audit.json"
    audit = json.loads(audit_path.read_text())
    pairs = audit["pairs"][:args.limit or None]
    names = [pair_name(pair) for pair in pairs]
    if not pairs or len(names) != len(set(names)):
        raise ValueError("empty or duplicate (mid,jid,seed) audit pairs")
    args.out.mkdir(parents=True, exist_ok=False)
    for method in METHODS:
        (args.out / method).mkdir()
    for directory in ("alignment", "alpha", "calibration"):
        (args.out / directory).mkdir()
    # Existing evaluator has three hardcoded controls. Explicit mapping, never
    # present these adapter directory names as the actual algorithms.
    aliases = {"feather_e4_f8": "inplace_feather", "feather_e8_f16": "warp_feather", "poisson_e4": "warp_multiband"}
    if args.dev_identity_layout:
        (args.out / "dev_identity_controls").mkdir()
        for alias in aliases:
            (args.out / "dev_identity_controls" / alias).mkdir()
        write_json(args.out / "dev_identity_controls" / "METHOD_MAP.json", aliases)
    selected_audit = {**audit, "pairs": pairs, "B_input_probe": str(args.probe.resolve())}
    write_json(args.out / "audit.json", selected_audit)
    # Small evaluator probe view: base points to files saved by this run.
    (args.out / "outputs").symlink_to("base", target_is_directory=True)
    manifest = {"formal_result": False, "config": asdict(config), "methods": list(METHODS),
                "method_config": {"base": {"operation": "unchanged base"},
                    "inplace_feather": {"alignment": "identity", "blend": "distance feather"},
                    "warp_feather": {"alignment": "bidirectional ECC translation", "blend": "distance feather"},
                    "warp_multiband": {"alignment": "same ECC translation", "blend": "Laplacian RGB / Gaussian alpha"}},
                "source_role": "M RGB and frozen M mask; base generated from M/I; no H read",
                "paired_boundary_ground_truth": {"available": False, "metrics": None},
                "metric_warning": "Source fidelity and boundary anomaly proxies are not paired-H fidelity or real defect rates. Synthetic known-clean scores are a separate calibration domain.",
                "sampling_convention": "output[y,x]=source[y+dy,x+dx]; visual source-to-output shift=(-dx,-dy)",
                "fallback": "rejected ECC => identity sampling, same fusion; status stays explicit",
                "support": "input M intersect valid source samples; alpha gates editing only, never ROI",
                "allowed_edit_region": "dilate(input M, square radius 8); also report stricter outside-M spill",
                "reliability_limit": "global ECC/FB/texture/support proxy; no local occlusion truth",
                "calibration_cases": calibration_cases(), "calibration_limit": "once per unique successfully loaded mid, M self-perturbations only",
                "script_sha256": sha256(__file__), "audit_sha256": sha256(audit_path),
                "opencv": cv2.__version__, "numpy": np.__version__,
                "args": {key: str(val) if isinstance(val, Path) else val for key, val in vars(args).items()},
                "dev_identity_method_map": aliases if args.dev_identity_layout else None}
    write_json(args.out / "method_config.json", manifest)
    start = time.monotonic()
    rows, alignment_rows, calibration_rows, calibrated_mids = [], [], [], set()
    with (args.out / "per_image.jsonl").open("x") as images, (args.out / "failures.jsonl").open("x") as failures, (args.out / "alignment.jsonl").open("x") as alignments, (args.out / "calibration.jsonl").open("x") as calibrations:
        for pair, name in zip(pairs, names):
            identity = {**pair, "name": name}
            try:
                if time.monotonic() - start > args.max_seconds:
                    raise TimeoutError("between_pair_budget_exhausted")
                spath = source_path(args.root, pair["mid"])
                bpath, mpath = args.probe / "outputs" / name, args.probe / "masks" / (safe_token(pair["mid"]) + ".png")
                source, base = read_rgb(spath), read_rgb(bpath)
                raw_mask = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
                if raw_mask is None:
                    raise ValueError("missing input mask")
                mask = raw_mask > 127
                if source.shape != base.shape or source.shape[:2] != mask.shape or not mask.any() or mask.all():
                    raise ValueError("shape mismatch or empty/full input mask; never resize or replace scoring ROI")
                context = reference_context(source, mask, config)
                inputs = {"source": str(spath), "source_sha256": sha256(spath), "base_sha256": sha256(bpath), "mask_sha256": sha256(mpath), "shape": list(source.shape)}
                alignment = estimate_alignment(source, base, mask, config)
                ar = {**identity, **alignment}
                alignment_rows.append(ar)
                emit_row(alignments, ar)
                if not alignment["accepted"]:
                    emit_row(failures, {**ar, "stage": "alignment", "identity_fallback_selected": True})
                write_json(args.out / "alignment" / (name + ".json"), ar)
            except Exception as exc:
                for method in METHODS:
                    row = {**identity, "method": method, "status": "failed", "stage": "input_or_budget", "error": str(exc)}
                    rows.append(row)
                    emit_row(images, row)
                    emit_row(failures, row)
                continue
            for method in METHODS:
                row = {**identity, "method": method, "status": "ok", "inputs": inputs,
                       "alignment_status": alignment["status"] if method.startswith("warp_") else "not_used"}
                method_start = time.monotonic()
                try:
                    output, alpha, coverage = method_output(method, source, base, mask, alignment, config)
                    row.update(metrics=measure(source, base, output, mask, config, context), coverage=coverage)
                    save_rgb(args.out / method / name, output)
                    if method != "base":
                        if not cv2.imwrite(str(args.out / "alpha" / (method + "__" + name)), np.rint(alpha * 255).astype(np.uint8)):
                            raise OSError("alpha PNG write failed")
                    if args.dev_identity_layout and method != "base":
                        alias = next(key for key, val in aliases.items() if val == method)
                        (args.out / "dev_identity_controls" / alias / name).hardlink_to(args.out / method / name)
                except Exception as exc:
                    row.update(status="failed", stage="method", error=str(exc))
                    emit_row(failures, row)
                row["seconds"] = time.monotonic() - method_start
                rows.append(row)
                emit_row(images, row)
            mid = str(pair["mid"])
            if mid not in calibrated_mids:
                try:
                    folder = args.out / "calibration" / safe_token(mid)
                    folder.mkdir()
                    local_rows = []
                    for kind, severity in calibration_cases():
                        perturbed = synthetic_perturbation(source, mask, kind, severity)
                        values = measure(source, source, perturbed, mask, config, context)
                        band = context["regions"]["boundary"]
                        cr = {"mid": mid, "kind": kind, "severity": severity,
                              "severity_unit": "u8_RGB_offset" if 'offset' in kind else "native_pixels" if 'misalignment' in kind else "none",
                              "boundary_area": int(band.sum()), "known_clean": "source M, not paired H",
                              **{key: values[key] for key in PROXIES},
                              "synthetic_boundary_rgb_mae_to_known_clean": float(np.abs(perturbed.astype(np.float32) - source)[band].mean() / 255),
                              "synthetic_transition_gradient_l1_to_known_clean": values['transition_gradient_l1_to_source'],
                              "synthetic_boundary_gradient_l1_to_known_clean": values["boundary_gradient_l1_to_source"]}
                        save_rgb(folder / f"{kind}_{severity}.png", perturbed)
                        local_rows.append(cr)
                    for cr in local_rows:
                        emit_row(calibrations, cr)
                    calibration_rows.extend(local_rows)
                    calibrated_mids.add(mid)
                except Exception as exc:
                    emit_row(failures, {**identity, "stage": "calibration", "error": str(exc)})
            print(json.dumps({"pairs_completed": len(rows) // len(METHODS), "expected": len(pairs),
                              "alignment": alignment["status"]}), flush=True)
    write_json(args.out / "calibration_summary.json", summarize_calibration(calibration_rows))
    accepted = sum(r["accepted"] for r in alignment_rows)
    failures_count = sum(r["status"] != "ok" for r in rows)
    summary = {"formal_result": False, "status": "complete" if not failures_count and len(calibrated_mids) == len({str(p["mid"]) for p in pairs}) else "complete_with_failures",
               "pairs_expected": len(pairs), "method_rows_expected": len(pairs) * len(METHODS),
               "method_rows_written": len(rows), "method_failures": failures_count,
               "alignment": {"attempted": len(alignment_rows), "accepted": accepted,
                              "identity_fallback": len(alignment_rows) - accepted,
                              "not_attempted": len(pairs) - len(alignment_rows),
                              "accepted_fraction_of_all_pairs": accepted / len(pairs)},
               "calibration_unique_M": len(calibrated_mids), "calibration_expected_unique_M": len({str(p["mid"]) for p in pairs}),
               "seconds": time.monotonic() - start, "gpu_hours": 0, "cpu_threads": args.threads,
               "budget": "soft between-pair check; ECC bounded to 2x60 iterations, no training",
               "methods": aggregate(rows), "paired_boundary_ground_truth": None,
               "warning": manifest["metric_warning"],
               "statistics": "exploratory descriptive means and paired deltas; failures remain in expected counts, valid means are conditional; no formal CI/gates"}
    write_json(args.out / "summary.json", summary)
    print(json.dumps({key: val for key, val in summary.items() if key != "methods"}), flush=True)
    return 0 if summary["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
