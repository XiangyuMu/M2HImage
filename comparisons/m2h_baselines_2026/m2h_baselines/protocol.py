from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Mapping

from .prepare import _prepare_one


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_counterfactual(
    *,
    data_root: Path,
    output_root: Path,
    protocol_path: Path,
    resolution: str,
    prep_config: Mapping[str, object],
    workers: int,
    limit_pairs: int | None = None,
) -> Path:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    pairs = list(protocol["pairs"])
    if limit_pairs is not None:
        pairs = pairs[:limit_pairs]
    unique_ids = sorted(
        {
            str(value)
            for pair in pairs
            for value in (pair["mannequin_id"], pair["identity_id"])
        }
    )
    jobs = [
        (sample_id, "counterfactual", resolution, str(data_root), str(output_root), dict(prep_config))
        for sample_id in unique_ids
    ]
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
    manifest_dir = output_root / resolution / "counterfactual"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    if failures:
        path = manifest_dir / "failures.json"
        path.write_text(json.dumps(failures, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        raise RuntimeError(f"counterfactual preparation failed for {len(failures)} IDs; see {path}")

    content_box = [64, 0, 448, 512] if resolution == "low" else [0, 0, 768, 1024]
    records: list[dict[str, object]] = []
    for pair_index, pair in enumerate(pairs):
        mid = str(pair["mannequin_id"])
        jid = str(pair["identity_id"])
        mid_paths = results[mid]["paths"]
        jid_paths = results[jid]["paths"]
        for seed in pair.get("seeds", [0]):
            sample_name = f"{pair_index:04d}_{mid}_{jid}_s{int(seed)}"
            records.append(
                {
                    "key": f"counterfactual:{sample_name}",
                    "sample_name": sample_name,
                    "split": "counterfactual",
                    "mid": mid,
                    "jid": jid,
                    "seed": int(seed),
                    "garment_type": pair.get("garment_type"),
                    "theta_source": pair.get("theta_source", mid),
                    "target": mid_paths["target"],
                    "mannequin": mid_paths["mannequin"],
                    "agnostic": mid_paths["agnostic"],
                    "replace_mask": mid_paths["replace_mask"],
                    "identity_card": jid_paths["identity_card"],
                    "face": jid_paths["face"],
                    "face_region": mid_paths["face_region"],
                    "garment": mid_paths["garment"],
                    "pose": mid_paths["pose"],
                    "resolution": resolution,
                    "content_box": content_box,
                }
            )
    manifest_path = manifest_dir / "samples.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = {
        "protocol": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "pairs": len(pairs),
        "records": len(records),
        "unique_ids": len(unique_ids),
        "resolution": resolution,
        "manifest_sha256": _sha256(manifest_path),
    }
    (manifest_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest_path
