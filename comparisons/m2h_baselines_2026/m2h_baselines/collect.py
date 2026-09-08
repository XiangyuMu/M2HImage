from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from PIL import Image

from .validate import load_manifest


EXPECTED_SIZE = {"low": (512, 512), "native": (768, 1024)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_name(record: dict[str, object]) -> str:
    return (
        f"{record['mid']}__id{record['jid']}__seed"
        f"{int(record.get('seed', 42))}.png"
    )


def protocol_from_records(records: Iterable[dict[str, object]]) -> dict[str, object]:
    records = list(records)
    pairs: dict[tuple[str, str], dict[str, object]] = {}
    mannequins: list[str] = []
    identities: list[str] = []
    garment_types: dict[str, str] = {}
    for record in records:
        mid, jid = str(record["mid"]), str(record["jid"])
        if mid not in mannequins:
            mannequins.append(mid)
        if jid not in identities:
            identities.append(jid)
        garment_type = str(record.get("garment_type") or "unknown")
        garment_types[mid] = garment_type
        key = (mid, jid)
        if key not in pairs:
            pairs[key] = {
                "mannequin_id": mid,
                "identity_id": jid,
                "theta_source": str(record.get("theta_source") or mid),
                "garment_type": garment_type,
                "seeds": [],
            }
        seed = int(record.get("seed", 42))
        seeds = pairs[key]["seeds"]
        if seed not in seeds:
            seeds.append(seed)
    for pair in pairs.values():
        pair["seeds"] = sorted(pair["seeds"])
    counts = Counter(garment_types.values())
    return {
        "seed": 42,
        "garment_type_source": "M2H frozen counterfactual manifest",
        "garment_type_counts": dict(sorted(counts.items())),
        "garment_types": garment_types,
        "mannequins": mannequins,
        "identity_pool": identities,
        "pairs": list(pairs.values()),
    }


def _source_index(raw_dir: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in raw_dir.rglob("*.png"):
        if path.is_file():
            index.setdefault(path.name, []).append(path)
    return index


def _link(source: Path, destination: Path, overwrite: bool) -> None:
    source = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() == source:
            return
        if not overwrite:
            raise FileExistsError(f"output link points elsewhere: {destination}")
        destination.unlink()
    elif destination.exists():
        if not overwrite:
            raise FileExistsError(f"refusing to replace output: {destination}")
        destination.unlink()
    destination.symlink_to(source)


def collect_outputs(
    *,
    manifest_path: Path,
    raw_dir: Path,
    output_dir: Path,
    method: str,
    checkpoint: str,
    overwrite: bool = False,
    limit: int | None = None,
) -> dict[str, object]:
    records = load_manifest(manifest_path)
    if not records:
        raise ValueError(f"empty manifest: {manifest_path}")
    if limit is not None:
        records = records[:limit]
    index = _source_index(raw_dir)
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for record in records:
        token = str(record.get("sample_name", record["mid"]))
        source_name = f"{token}.png"
        candidates = index.get(source_name, [])
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"expected exactly one {source_name} under {raw_dir}, found {len(candidates)}"
            )
        source = candidates[0]
        with Image.open(source) as image:
            image.load()
            expected = EXPECTED_SIZE[str(record["resolution"])]
            if image.size != expected:
                raise ValueError(f"{source}: size={image.size}, expected={expected}")
        name = canonical_name(record)
        if name in seen:
            raise ValueError(f"duplicate canonical output: {name}")
        seen.add(name)
        destination = output_dir / name
        _link(source, destination, overwrite)
        rows.append(
            {
                "key": record["key"],
                "mid": record["mid"],
                "jid": record["jid"],
                "seed": int(record.get("seed", 42)),
                "method": method,
                "checkpoint": checkpoint,
                "source": str(source.resolve()),
                "output": str(destination.resolve()),
                "sha256": sha256(source),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    inference_manifest = output_dir / "inference_manifest.jsonl"
    inference_manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    protocol_path = output_dir / "protocol.json"
    protocol_path.write_text(
        json.dumps(protocol_from_records(records), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary = {
        "status": "pass",
        "method": method,
        "checkpoint": checkpoint,
        "records": len(rows),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path),
        "inference_manifest": str(inference_manifest),
        "inference_manifest_sha256": sha256(inference_manifest),
        "protocol": str(protocol_path),
        "raw_dir": str(raw_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect baseline outputs for M2H Metrics-v2.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    summary = collect_outputs(
        manifest_path=Path(args.manifest),
        raw_dir=Path(args.raw_dir),
        output_dir=Path(args.output_dir),
        method=args.method,
        checkpoint=args.checkpoint,
        overwrite=args.overwrite,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
