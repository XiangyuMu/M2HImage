from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from m2h_baselines.nativeize import nativeize_outputs


def test_nativeize_crops_content_box_and_writes_provenance(tmp_path: Path) -> None:
    low = tmp_path / "low"
    native = tmp_path / "native_manifest.jsonl"
    output = tmp_path / "native"
    low.mkdir()
    # The content region is red; the two 64px borders are blue.
    image = Image.new("RGB", (512, 512), (0, 0, 255))
    for x in range(64, 448):
        for y in range(512):
            image.putpixel((x, y), (255, 0, 0))
    name = "0000_mid_jid_s0"
    canonical = "mid__idjid__seed0.png"
    image.save(low / canonical)
    record = {
        "key": "counterfactual:" + name,
        "sample_name": name,
        "mid": "mid",
        "jid": "jid",
        "seed": 0,
        "resolution": "native",
        "target": "native/counterfactual/target/mid.png",
    }
    native.write_text(json.dumps(record) + "\n", encoding="utf-8")

    summary = nativeize_outputs(
        low_canonical=low,
        native_manifest=native,
        output_dir=output,
        method="test_method",
        checkpoint="/ckpt",
    )

    assert summary["records"] == 1
    with Image.open(output / canonical) as converted:
        assert converted.size == (768, 1024)
        assert converted.getpixel((0, 0))[0] > 240
        assert converted.getpixel((0, 0))[2] < 20
    rows = [json.loads(line) for line in (output / "inference_manifest.jsonl").read_text().splitlines()]
    assert rows[0]["method"] == "test_method"
    assert rows[0]["checkpoint"] == "/ckpt"
