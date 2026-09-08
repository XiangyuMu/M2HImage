from __future__ import annotations

import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from .spec import SampleSpec


HEAD_EDGES = ((1, 0), (0, 2), (3, 1), (2, 4))
HEAD_COLORS = ((255, 80, 80), (255, 180, 60), (80, 220, 255), (180, 100, 255))


def find_one(directory: Path, sample_id: str) -> Path:
    for suffix in (".png", ".jpg", ".jpeg", ".webp"):
        path = directory / f"{sample_id}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"missing image for {sample_id} under {directory}")


def read_ids(path: Path) -> list[str]:
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ids.append(Path(line.split()[0]).stem)
    return ids


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _letterbox(image: Image.Image, is_mask: bool = False) -> Image.Image:
    resampling = Image.Resampling.NEAREST if is_mask else Image.Resampling.LANCZOS
    resized = image.resize((384, 512), resampling)
    array = np.asarray(resized)
    if array.ndim == 2:
        padded = np.pad(array, ((0, 0), (64, 64)), mode="constant", constant_values=0)
        return Image.fromarray(padded.astype(np.uint8), mode="L")
    padded = np.pad(array, ((0, 0), (64, 64), (0, 0)), mode="reflect")
    return Image.fromarray(padded.astype(np.uint8), mode="RGB")


def _resize(image: Image.Image, resolution: str, is_mask: bool = False) -> Image.Image:
    image = image.convert("L" if is_mask else "RGB")
    if resolution == "low":
        return _letterbox(image, is_mask=is_mask)
    if resolution == "native":
        return image.resize(
            (768, 1024),
            Image.Resampling.NEAREST if is_mask else Image.Resampling.LANCZOS,
        )
    raise ValueError(f"unknown resolution: {resolution}")


def _bbox(mask: np.ndarray, padding: float) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return None
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    pad = int(round(max(x1 - x0, y1 - y0) * padding))
    return max(0, x0 - pad), max(0, y0 - pad), min(mask.shape[1], x1 + pad), min(mask.shape[0], y1 + pad)


def _masked_card(
    image: Image.Image,
    mask: np.ndarray,
    padding: float,
    fallback: Image.Image | None = None,
) -> Image.Image:
    box = _bbox(mask, padding)
    if box is None:
        if fallback is None:
            raise RuntimeError("empty card mask and no fallback")
        crop = fallback.convert("RGB")
    else:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        neutral = np.full_like(rgb, 127)
        alpha = (mask.astype(np.float32) / 255.0)[..., None]
        isolated = (rgb * alpha + neutral * (1.0 - alpha)).astype(np.uint8)
        crop = Image.fromarray(isolated, mode="RGB").crop(box)
    return ImageOps.pad(crop, (512, 512), method=Image.Resampling.LANCZOS, color=(127, 127, 127))


def _synth_head_points(theta: Mapping[str, float], neck: Sequence[float], scale: float, offset: float) -> np.ndarray:
    yaw, pitch, roll = (math.radians(float(theta.get(key, 0.0))) for key in ("yaw", "pitch", "roll"))
    points = np.asarray(
        [[0.0, 0.02], [-0.22, -0.18], [0.22, -0.18], [-0.48, -0.04], [0.48, -0.04]],
        dtype=np.float32,
    )
    points[:, 0] *= max(0.20, math.cos(yaw))
    points[0, 0] += 0.24 * math.sin(yaw)
    points[:, 1] += 0.12 * math.sin(pitch)
    rotation = np.asarray(
        [[math.cos(roll), -math.sin(roll)], [math.sin(roll), math.cos(roll)]], dtype=np.float32
    )
    points = points @ rotation.T
    anchor = np.asarray(neck, dtype=np.float32) - np.asarray([0.0, offset * scale], dtype=np.float32)
    return points * scale + anchor


def _pose_with_head(data_root: Path, sample_id: str, nose_offset: float) -> Image.Image:
    pose = Image.open(find_one(data_root / "dwpose/without_head/mannequin", sample_id)).convert("RGB")
    pose = pose.resize((768, 1024), Image.Resampling.BICUBIC)
    kp_path = data_root / "dwpose/keypoints/mannequin" / f"{sample_id}.npz"
    hp_path = data_root / "derived/head_pose_6drepnet/human" / f"{sample_id}.json"
    if not kp_path.exists() or not hp_path.exists():
        return pose
    with np.load(kp_path, allow_pickle=False) as payload:
        body = np.asarray(payload["body"], dtype=np.float32)
        scores = np.asarray(payload["body_scores"], dtype=np.float32)
    if body.shape != (18, 2) or scores.shape != (18,):
        return pose
    theta_payload = json.loads(hp_path.read_text(encoding="utf-8"))
    theta = theta_payload if theta_payload.get("status") == "ok" else {"yaw": 0, "pitch": 0, "roll": 0}
    neck = body[1] * np.asarray([768, 1024], dtype=np.float32)
    shoulder = abs(float(body[2, 0] - body[5, 0])) * 768 if scores[2] >= 0.3 and scores[5] >= 0.3 else 0.22 * 768
    points = _synth_head_points(theta, neck, max(24.0, 0.48 * shoulder), nose_offset)
    draw = ImageDraw.Draw(pose)
    for edge, color in zip(HEAD_EDGES, HEAD_COLORS, strict=True):
        left, right = points[edge[0]], points[edge[1]]
        draw.line((float(left[0]), float(left[1]), float(right[0]), float(right[1])), fill=color, width=3)
    for point in points:
        x, y = map(float, point)
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(255, 255, 255))
    return pose


def _condition_masks(data_root: Path, sample_id: str, cfg: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray]:
    parsing = np.asarray(
        Image.open(find_one(data_root / "human_parsing/fashn/masks/mannequin", sample_id)).convert("L"),
        dtype=np.uint8,
    )
    replace = np.isin(parsing, np.asarray(cfg["replace_labels"], dtype=np.uint8)).astype(np.uint8) * 255
    with np.load(data_root / "derived/region_masks" / f"{sample_id}.npz", allow_pickle=False) as payload:
        cloth = np.asarray(payload["cloth"], dtype=np.uint8)
        aligned_identity = np.maximum(
            np.asarray(payload["id_strong"], dtype=np.uint8),
            np.asarray(payload["id_weak"], dtype=np.uint8),
        )
    replace = np.maximum(replace, aligned_identity)
    replace_image = Image.fromarray(replace, mode="L").filter(ImageFilter.MaxFilter(int(cfg["mask_dilation"])))
    cloth_guard = Image.fromarray(cloth, mode="L").filter(ImageFilter.MaxFilter(int(cfg["cloth_protect_dilation"])))
    replace = np.where(np.asarray(cloth_guard) >= 128, 0, np.asarray(replace_image)).astype(np.uint8)
    replace = np.where(replace >= 128, 255, 0).astype(np.uint8)
    return replace, cloth


def _save(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False)


def _prepare_one(args: tuple[str, str, str, str, str, dict[str, object]]) -> dict[str, object]:
    sample_id, split, resolution, data_root_raw, output_root_raw, prep_cfg = args
    data_root, output_root = Path(data_root_raw), Path(output_root_raw)
    human = Image.open(find_one(data_root / "images/human", sample_id)).convert("RGB")
    mannequin = Image.open(find_one(data_root / "images/mannequin", sample_id)).convert("RGB")
    face = Image.open(find_one(data_root / "derived/face_crops/human", sample_id)).convert("RGB")
    replace, cloth = _condition_masks(data_root, sample_id, prep_cfg)

    neutral = Image.new("RGB", mannequin.size, tuple(int(x) for x in prep_cfg["neutral_rgb"]))
    feather = Image.fromarray(replace, mode="L").filter(ImageFilter.GaussianBlur(radius=2.0))
    agnostic = Image.composite(neutral, mannequin, feather)
    with np.load(data_root / "derived/region_masks" / f"{sample_id}.npz", allow_pickle=False) as payload:
        identity_mask = np.asarray(payload["id_strong"], dtype=np.uint8)
        identity_person_mask = np.maximum(
            identity_mask,
            np.asarray(payload["id_weak"], dtype=np.uint8),
        )
    identity_card = _masked_card(human, identity_mask, float(prep_cfg["identity_crop_padding"]), face)
    identity_person = _masked_card(
        human,
        identity_person_mask,
        float(prep_cfg.get("identity_person_crop_padding", 0.12)),
        face,
    )
    garment_card = _masked_card(mannequin, cloth, float(prep_cfg["garment_crop_padding"]))
    pose = _pose_with_head(data_root, sample_id, float(prep_cfg["head_nose_offset_scale"]))
    pose_dense = Image.open(
        find_one(data_root / "dwpose/with_head/mannequin", sample_id)
    ).convert("RGB")

    # This method-specific condition is intentionally not added to the frozen
    # sample manifest. It is resolved by materializers from split/id so adding
    # it cannot invalidate already collected baseline manifests.
    identity_person_path = Path(resolution) / split / "identity_person" / f"{sample_id}.png"
    _save(identity_person, output_root / identity_person_path)
    pose_dense_path = Path(resolution) / split / "pose_dense" / f"{sample_id}.png"
    _save(_resize(pose_dense, resolution), output_root / pose_dense_path)

    images = {
        "target": _resize(human, resolution),
        "mannequin": _resize(mannequin, resolution),
        "agnostic": _resize(agnostic, resolution),
        "replace_mask": _resize(Image.fromarray(replace, mode="L"), resolution, is_mask=True),
        "identity_card": identity_card,
        "face": ImageOps.pad(face, (512, 512), method=Image.Resampling.LANCZOS, color=(127, 127, 127)),
        "face_region": _resize(Image.fromarray(identity_mask, mode="L"), resolution, is_mask=True),
        "garment": garment_card,
        "pose": _resize(pose, resolution),
    }
    relative: dict[str, str] = {}
    for field, image in images.items():
        relative_path = Path(resolution) / split / field / f"{sample_id}.png"
        _save(image, output_root / relative_path)
        relative[field] = str(relative_path)
    return {"id": sample_id, "paths": relative}


def _iter_records(ids: Iterable[str], split: str, resolution: str, result_by_id: Mapping[str, dict[str, object]]) -> Iterable[SampleSpec]:
    content_box = (64, 0, 448, 512) if resolution == "low" else (0, 0, 768, 1024)
    for sample_id in ids:
        paths = result_by_id[sample_id]["paths"]
        yield SampleSpec(
            key=f"{split}:{sample_id}:{sample_id}",
            split=split,
            mid=sample_id,
            jid=sample_id,
            resolution=resolution,
            content_box=content_box,
            **paths,
        )


def prepare_split(
    *,
    data_root: Path,
    output_root: Path,
    split: str,
    resolution: str,
    prep_config: Mapping[str, object],
    workers: int,
    limit: int | None = None,
    excluded_ids: Sequence[str] = (),
) -> Path:
    split_file = data_root / "splits" / f"{split}.txt"
    ids = [sample_id for sample_id in read_ids(split_file) if sample_id not in set(excluded_ids)]
    if limit is not None:
        ids = ids[:limit]
    output_root.mkdir(parents=True, exist_ok=True)
    jobs = [(sample_id, split, resolution, str(data_root), str(output_root), dict(prep_config)) for sample_id in ids]
    results: dict[str, dict[str, object]] = {}
    failures: list[dict[str, str]] = []
    with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(_prepare_one, job): job[0] for job in jobs}
        for future in as_completed(futures):
            sample_id = futures[future]
            try:
                results[sample_id] = future.result()
            except Exception as exc:  # noqa: BLE001
                failures.append({"id": sample_id, "error": str(exc)})
    manifest_path = output_root / resolution / split / "samples.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if failures:
        failure_path = manifest_path.with_name("failures.json")
        failure_path.write_text(json.dumps(failures, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        raise RuntimeError(f"failed to prepare {len(failures)} samples; see {failure_path}")
    records = list(_iter_records(ids, split, resolution, results))
    manifest_path.write_text("".join(json.dumps(record.to_dict(), ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    summary = {
        "split": split,
        "resolution": resolution,
        "count": len(records),
        "source_split_sha256": _sha256(split_file),
        "excluded_ids": sorted(set(excluded_ids)),
        "content_box": list(records[0].content_box) if records else None,
        "manifest_sha256": _sha256(manifest_path),
    }
    manifest_path.with_name("summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest_path
