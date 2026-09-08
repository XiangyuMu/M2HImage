from __future__ import annotations

"""Convert square model outputs back to the frozen native M2H canvas.

The current VTON implementations train and sample at 512x512.  The M2H
protocol also defines a native 768x1024 view whose model content occupies the
central 384x512 region of the square staging canvas.  This module performs the
deterministic inverse transform and writes a self-contained canonical bundle.
"""

import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image

from .collect import canonical_name, protocol_from_records
from .validate import load_manifest


LOW_SIZE = (512, 512)
NATIVE_SIZE = (768, 1024)
DEFAULT_LOW_CONTENT_BOX = (64, 0, 448, 512)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _native_image(source: Path, destination: Path, low_content_box: tuple[int, int, int, int]) -> None:
    with Image.open(source) as image:
        image.load()
        if image.size != LOW_SIZE:
            raise ValueError(f"{source}: size={image.size}, expected={LOW_SIZE}")
        crop = image.convert("RGB").crop(low_content_box)
        if crop.size != (384, 512):
            raise ValueError(f"invalid low content box {low_content_box}: crop={crop.size}")
        output = crop.resize(NATIVE_SIZE, Image.Resampling.LANCZOS)
        destination.parent.mkdir(parents=True, exist_ok=True)
        output.save(destination, format="PNG", optimize=False)


def nativeize_outputs(
    *,
    low_canonical: Path,
    native_manifest: Path,
    output_dir: Path,
    method: str,
    checkpoint: str,
    low_content_box: tuple[int, int, int, int] = DEFAULT_LOW_CONTENT_BOX,
    overwrite: bool = False,
) -> dict[str, object]:
    checkpoint = str(Path(checkpoint).expanduser().resolve())
    records = load_manifest(native_manifest)
    if not records:
        raise ValueError(f"empty native manifest: {native_manifest}")
    if low_canonical.resolve() == output_dir.resolve():
        raise ValueError("low canonical and native output directories must differ")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for record in records:
        name = canonical_name(record)
        if name in seen:
            raise ValueError(f"duplicate canonical name: {name}")
        seen.add(name)
        source_low = low_canonical / name
        if not source_low.is_file():
            raise FileNotFoundError(f"missing low canonical output: {source_low}")
        destination = output_dir / name
        if destination.exists() and not overwrite:
            raise FileExistsError(f"refusing to replace existing native output: {destination}")
        if destination.is_symlink():
            destination.unlink()
        _native_image(source_low, destination, low_content_box)
        rows.append(
            {
                "key": record["key"],
                "sample_name": record.get("sample_name"),
                "mid": record["mid"],
                "jid": record["jid"],
                "seed": int(record.get("seed", 42)),
                "method": method,
                "checkpoint": checkpoint,
                "source_low": str(source_low.resolve()),
                "source": str(destination.resolve()),
                "output": str(destination.resolve()),
                "sha256": sha256(destination),
                "resolution": "native",
            }
        )

    inference_manifest = output_dir / "inference_manifest.jsonl"
    inference_manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    protocol_path = output_dir / "protocol.json"
    _write_json(protocol_path, protocol_from_records(records))
    summary = {
        "status": "pass",
        "method": method,
        "checkpoint": checkpoint,
        "records": len(rows),
        "resolution": "native",
        "size": list(NATIVE_SIZE),
        "manifest": str(native_manifest.resolve()),
        "manifest_sha256": sha256(native_manifest),
        "source_low_canonical": str(low_canonical.resolve()),
        "low_content_box": list(low_content_box),
        "inference_manifest": str(inference_manifest.resolve()),
        "inference_manifest_sha256": sha256(inference_manifest),
        "protocol": str(protocol_path.resolve()),
        "output_dir": str(output_dir.resolve()),
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def _parse_box(value: str) -> tuple[int, int, int, int]:
    values = tuple(int(item.strip()) for item in value.split(","))
    if len(values) != 4:
        raise argparse.ArgumentTypeError("content box must contain x0,y0,x1,y1")
    x0, y0, x1, y1 = values
    if not (0 <= x0 < x1 <= LOW_SIZE[0] and 0 <= y0 < y1 <= LOW_SIZE[1]):
        raise argparse.ArgumentTypeError(f"content box outside {LOW_SIZE}: {values}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert 512x512 M2H outputs to 768x1024 native canonical outputs.")
    parser.add_argument("--low-canonical", required=True, type=Path)
    parser.add_argument("--native-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--method", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--low-content-box", type=_parse_box, default=DEFAULT_LOW_CONTENT_BOX)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = nativeize_outputs(
        low_canonical=args.low_canonical,
        native_manifest=args.native_manifest,
        output_dir=args.output_dir,
        method=args.method,
        checkpoint=args.checkpoint,
        low_content_box=args.low_content_box,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
