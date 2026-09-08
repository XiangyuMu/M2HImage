#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_name(row: dict[str, object]) -> str:
    return f"{row['mid']}__id{row['jid']}__seed{int(row['seed'])}.png"


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strictly verify one canonical M2H baseline inference run."
    )
    parser.add_argument("--canonical", required=True, type=Path)
    parser.add_argument("--method", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--records", type=int, default=4)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--frozen-protocol", required=True, type=Path)
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()

    canonical = args.canonical.resolve()
    summary_path = canonical / "summary.json"
    summary = load_json(summary_path)
    expected_summary = {
        "status": "pass",
        "method": args.method,
        "checkpoint": args.checkpoint,
        "records": args.records,
    }
    for key, expected in expected_summary.items():
        actual = summary.get(key)
        if actual != expected:
            raise RuntimeError(
                f"summary {key} mismatch: {actual!r} != {expected!r}"
            )

    manifest_path = Path(str(summary["manifest"]))
    if sha256(manifest_path) != summary.get("manifest_sha256"):
        raise RuntimeError("prepared manifest SHA256 does not match summary.json")

    inference_manifest = Path(str(summary["inference_manifest"]))
    if inference_manifest.resolve() != (canonical / "inference_manifest.jsonl"):
        raise RuntimeError("summary points to an unexpected inference manifest")
    if sha256(inference_manifest) != summary.get("inference_manifest_sha256"):
        raise RuntimeError("inference manifest SHA256 does not match summary.json")

    rows = [
        json.loads(line)
        for line in inference_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != args.records:
        raise RuntimeError(f"inference rows mismatch: {len(rows)} != {args.records}")

    expected_names = {canonical_name(row) for row in rows}
    if len(expected_names) != args.records:
        raise RuntimeError("canonical names are not unique")
    actual_names = {path.name for path in canonical.glob("*.png")}
    if actual_names != expected_names:
        raise RuntimeError(
            f"canonical PNG set mismatch: actual={sorted(actual_names)}, "
            f"expected={sorted(expected_names)}"
        )

    image_hashes: dict[str, str] = {}
    for row in rows:
        if row.get("method") != args.method:
            raise RuntimeError(f"row method mismatch: {row}")
        if row.get("checkpoint") != args.checkpoint:
            raise RuntimeError(f"row checkpoint mismatch: {row}")
        name = canonical_name(row)
        image_path = canonical / name
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as image:
            image.load()
            if image.size != (args.width, args.height):
                raise RuntimeError(
                    f"{image_path}: size={image.size}, "
                    f"expected={(args.width, args.height)}"
                )
        image_sha = sha256(image_path)
        if image_sha != row.get("sha256"):
            raise RuntimeError(f"{image_path}: SHA256 mismatch")
        source = Path(str(row["source"])).resolve()
        if image_path.resolve() != source:
            raise RuntimeError(f"{image_path}: does not resolve to {source}")
        image_hashes[name] = image_sha

    protocol_path = Path(str(summary["protocol"]))
    protocol = load_json(protocol_path)
    protocol_pairs = protocol.get("pairs")
    if not isinstance(protocol_pairs, list) or not protocol_pairs:
        raise RuntimeError("generated protocol has no pairs")

    frozen_protocol_sha = sha256(args.frozen_protocol)
    if frozen_protocol_sha != args.protocol_sha256:
        raise RuntimeError(
            "frozen protocol SHA256 mismatch: "
            f"{frozen_protocol_sha} != {args.protocol_sha256}"
        )

    report = {
        "status": "pass",
        "method": args.method,
        "checkpoint": args.checkpoint,
        "records": len(rows),
        "size": [args.width, args.height],
        "canonical_names": sorted(expected_names),
        "image_sha256": dict(sorted(image_hashes.items())),
        "inference_manifest_sha256": sha256(inference_manifest),
        "frozen_protocol_sha256": frozen_protocol_sha,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
