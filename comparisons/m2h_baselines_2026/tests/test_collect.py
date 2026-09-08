from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image

from m2h_baselines.collect import collect_outputs
from m2h_baselines.metrics_config import write_metrics_config


class CollectTest(unittest.TestCase):
    def test_collects_canonical_names_and_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "samples.jsonl"
            raw = root / "raw" / "nested"
            canonical = root / "canonical"
            raw.mkdir(parents=True)
            records = []
            for seed in (0, 1):
                token = f"0000_00001_00002_s{seed}"
                record = {
                    "key": f"counterfactual:{token}",
                    "sample_name": token,
                    "split": "counterfactual",
                    "mid": "00001",
                    "jid": "00002",
                    "seed": seed,
                    "garment_type": "top",
                    "theta_source": "00001",
                    "resolution": "low",
                }
                records.append(record)
                Image.new("RGB", (512, 512), (seed, 20, 30)).save(raw / f"{token}.png")
            manifest.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            report = collect_outputs(
                manifest_path=manifest,
                raw_dir=root / "raw",
                output_dir=canonical,
                method="refton",
                checkpoint="checkpoint-100",
            )
            self.assertEqual(report["records"], 2)
            self.assertTrue((canonical / "00001__id00002__seed0.png").is_symlink())
            protocol = json.loads((canonical / "protocol.json").read_text(encoding="utf-8"))
            self.assertEqual(protocol["pairs"][0]["seeds"], [0, 1])

    def test_metrics_config_uses_manifest_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = root / "metrics.yaml"
            base.write_text("metrics_v2: {}\n", encoding="utf-8")
            manifest = root / "samples.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "key": "counterfactual:x",
                        "mid": "1",
                        "jid": "2",
                        "seed": 0,
                        "resolution": "low",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config, protocol = write_metrics_config(
                base_config=base,
                manifest=manifest,
                gen_dir=root / "generated",
                data_root=root / "dataset",
                output_config=root / "out" / "config.yaml",
                output_root=root / "report",
                run_name="mcld_smoke",
                label="MCLD smoke",
            )
            payload = yaml.safe_load(config.read_text(encoding="utf-8"))
            self.assertEqual(payload["data"]["resolution"], {"width": 512, "height": 512})
            self.assertTrue(protocol.is_file())


if __name__ == "__main__":
    unittest.main()
