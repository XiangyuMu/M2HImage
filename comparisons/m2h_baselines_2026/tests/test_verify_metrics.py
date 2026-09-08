import csv
import json
import tempfile
import unittest
from pathlib import Path

from m2h_baselines.verify_metrics import verify_metrics_outputs


class VerifyMetricsTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path]:
        run_dir = root / "run"
        canonical_dir = root / "canonical"
        masks_dir = run_dir / "parsing_masks"
        identity_dir = run_dir / "identity"
        canonical_dir.mkdir()
        masks_dir.mkdir(parents=True)
        identity_dir.mkdir()
        keys = [
            "01968__id35295__seed0",
            "01968__id35295__seed1",
            "01968__id18372__seed0",
            "01968__id18372__seed1",
        ]
        for key in keys:
            (canonical_dir / f"{key}.png").write_bytes(b"png")
            (masks_dir / f"{key}.png").write_bytes(b"mask")

        def dump(name: str, payload: dict[str, object]) -> None:
            (run_dir / name).write_text(json.dumps(payload))

        dump("parsing_summary.json", {"expected": 4, "cached": 4, "computed": 0, "failed": []})
        dump("garment_summary.json", {"status": "ok", "count": 4, "failed": 0})
        dump("hair_summary.json", {"status": "ok", "count": 4, "failed": 0, "no_hair": 0})
        dump("pose_summary.json", {"status": "ok", "count": 4, "failed": 0})
        dump(
            "identity_summary.json",
            {"status": "ok", "count": 2, "failed": 2, "face_detection_rate": 0.5},
        )

        rows = []
        for key in keys:
            mid, jid, seed = key.split("__")
            rows.append(
                {
                    "mid": mid,
                    "jid": jid.removeprefix("id"),
                    "seed": seed.removeprefix("seed"),
                    "status": "ok",
                }
            )
        csv_paths = {
            "garment": run_dir / "garment_per_image.csv",
            "hair": run_dir / "hair_per_image.csv",
            "pose": run_dir / "pose_per_image.csv",
            "identity": identity_dir / "deltaid_per_image.csv",
        }
        for label, path in csv_paths.items():
            output_rows = [dict(row) for row in rows]
            if label == "identity":
                output_rows[0]["status"] = "no_face"
                output_rows[1]["status"] = "no_face"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("mid", "jid", "seed", "status"))
                writer.writeheader()
                writer.writerows(output_rows)

        contact_sheet = run_dir / "contact_sheet.png"
        panel_index = run_dir / "panel_index.csv"
        report = run_dir / "report.md"
        contact_sheet.write_bytes(b"panel")
        panel_index.write_text("case\n1\n")
        report.write_text("report")
        dump(
            "panels_summary.json",
            {"case_count": 2, "contact_sheet": str(contact_sheet), "index_csv": str(panel_index)},
        )
        dump("report_bundle.json", {"report": str(report)})
        return run_dir, canonical_dir

    def test_quality_failures_are_strict_by_default_but_can_be_nonblocking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir, canonical_dir = self._fixture(Path(directory))
            with self.assertRaisesRegex(RuntimeError, "identity face detection rate"):
                verify_metrics_outputs(run_dir, canonical_dir, "smoke")

            result = verify_metrics_outputs(
                run_dir,
                canonical_dir,
                "smoke",
                allow_quality_failures=True,
            )
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["quality_status"], "warning")
            self.assertEqual(result["errors"], [])
            self.assertTrue(any("face detection rate" in item for item in result["quality_warnings"]))


if __name__ == "__main__":
    unittest.main()
