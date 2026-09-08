import json
import tempfile
import unittest
from pathlib import Path

from summarize_revalidation import (
    ARMS,
    METRIC_FIELDS,
    compute_effects,
    main,
    paired_delta,
    validate_rows,
)


def row(mid, jid, seed, branch, **metrics):
    base = {
        "mid": mid,
        "jid": jid,
        "seed": seed,
        "branch": branch,
        "tau": 0,
        "k": 1,
        "m_source_group": f"sg-{mid}",
        "path": f"{branch}/{mid}_{jid}_{seed}.png",
    }
    for field in METRIC_FIELDS:
        base[field] = metrics.get(field, 1.0)
    return base


def full_rows(delta=2.0):
    rows = []
    for i in range(128):
        mid = f"m{i:03d}"
        jid = f"j{i % 16:03d}"
        seed = 7
        values = {
            "b2_legacyM_freshI": 10.0,
            "a4_legacyM_freshI": 10.0 + delta,
            "b2_inputOnly_freshI": 20.0,
            "a4_inputOnly_freshI": 20.0 + delta + 3.0,
        }
        for arm in ARMS:
            rows.append(row(mid, jid, seed, arm, id_penalized=values[arm]))
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for item in rows:
            f.write(json.dumps(item) + "\n")


class SummarizeRevalidationTests(unittest.TestCase):
    def test_validate_requires_common_128_pair_keys(self):
        rows = full_rows()
        validate_rows(rows)
        bad = rows[:-1]
        with self.assertRaisesRegex(ValueError, "128"):
            validate_rows(bad)

    def test_missing_key_is_rejected(self):
        rows = full_rows()
        del rows[0]["jid"]
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_rows(rows)

    def test_missing_critical_metric_blocks_ready(self):
        rows = full_rows()
        rows[0]["garment_dino"] = None
        with self.assertRaisesRegex(ValueError, "critical metric None"):
            validate_rows(rows)

    def test_nonfinite_metric_blocks_early(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            metric_rows = [{**item, "metric_status": "ok"} for item in full_rows()]
            metric_rows[0] = {**metric_rows[0], "id_penalized": float("inf")}
            write_jsonl(run / "metrics_image" / "per_image.jsonl", metric_rows)
            write_jsonl(run / "metrics_id" / "per_image.jsonl", [{**item, "metric_status": "ok"} for item in full_rows()])
            write_jsonl(run / "metrics_pose" / "per_image.jsonl", [{**item, "metric_status": "ok"} for item in full_rows()])
            with self.assertRaisesRegex(ValueError, "non-numeric metric"):
                main(["--run", str(run), "--out", str(root / "out")])
            self.assertFalse((root / "out" / "READY").exists())

    def test_infrastructure_failure_blocks_cli_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            metric_rows = [{**item, "metric_status": "ok"} for item in full_rows()]
            failed_rows = list(metric_rows)
            failed_rows[0] = {**failed_rows[0], "metric_status": "reference_failed"}
            write_jsonl(run / "metrics_image" / "per_image.jsonl", metric_rows)
            write_jsonl(run / "metrics_id" / "per_image.jsonl", failed_rows)
            write_jsonl(run / "metrics_pose" / "per_image.jsonl", metric_rows)
            with self.assertRaisesRegex(ValueError, "infrastructure"):
                main(["--run", str(run), "--out", str(root / "out"), "--bootstrap-draws", "10"])
            self.assertFalse((root / "out" / "READY").exists())

    def test_paired_bootstrap_constant_difference(self):
        left = [row(f"m{i}", f"j{i % 2}", 0, "a4_legacyM_freshI", id_penalized=3.0) for i in range(8)]
        right = [row(f"m{i}", f"j{i % 2}", 0, "b2_legacyM_freshI", id_penalized=1.0) for i in range(8)]
        result = paired_delta(left, right, "id_penalized", draws=200, seed=20260907)
        self.assertEqual(result["delta_mean"], 2.0)
        self.assertEqual(result["ci95"], [2.0, 2.0])
        self.assertEqual(result["bootstrap"]["source_groups"], 8)
        self.assertEqual(result["bootstrap"]["reference_files"], 2)

    def test_a4_minus_b2_and_did_direction(self):
        effects = compute_effects(full_rows(delta=2.0), draws=200, seed=20260907)
        self.assertEqual(effects["a4_minus_b2"]["legacyM_freshI"]["id_penalized"]["delta_mean"], 2.0)
        self.assertEqual(effects["a4_minus_b2"]["inputOnly_freshI"]["id_penalized"]["delta_mean"], 5.0)
        self.assertEqual(effects["difference_in_differences"]["id_penalized"]["delta_mean"], 3.0)

    def test_cli_outputs_ready_and_ignores_carryin(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            metric_rows = []
            for item in full_rows():
                metric_rows.append({**item, "metric_status": "ok", "carryin_id_penalized": 999})
            write_jsonl(run / "metrics_image" / "per_image.jsonl", metric_rows)
            write_jsonl(run / "metrics_id" / "per_image.jsonl", metric_rows)
            write_jsonl(run / "metrics_pose" / "per_image.jsonl", metric_rows)
            manifest = {
                "pairs": [
                    {"mid": f"m{i:03d}", "jid": f"j{i % 16:03d}", "m_source_group": f"sg-m{i:03d}"}
                    for i in range(128)
                ]
            }
            (root / "manifest.json").write_text(json.dumps(manifest))
            main([
                "--run",
                str(run),
                "--out",
                str(root / "out"),
                "--manifest-json",
                str(root / "manifest.json"),
                "--bootstrap-draws",
                "200",
            ])
            self.assertTrue((root / "out" / "READY").is_file())
            summary = json.loads((root / "out" / "summary.json").read_text())
            self.assertNotIn("carryin_id_penalized", summary["metric_fields"])
            first = json.loads((root / "out" / "rows.jsonl").read_text().splitlines()[0])
            self.assertNotIn("carryin_id_penalized", first)

    def test_cli_uses_manifest_sensitivity_pair_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            metric_rows = [{**item, "metric_status": "ok"} for item in full_rows()]
            write_jsonl(run / "metrics_image" / "per_image.jsonl", metric_rows)
            write_jsonl(run / "metrics_id" / "per_image.jsonl", metric_rows)
            write_jsonl(run / "metrics_pose" / "per_image.jsonl", metric_rows)
            pairs = [
                {
                    "pair_index": i,
                    "mid": f"m{i:03d}",
                    "jid": f"j{i % 16:03d}",
                    "m_source_group": f"sg-m{i:03d}",
                    "near_duplicate_sensitivity_remove": i >= 101,
                }
                for i in range(128)
            ]
            protocol_manifest = {
                "schema": "m2h_minimal_revalidation_protocol_v1",
                "pairs": pairs,
                "sensitivity_pair_indices": list(range(101)),
            }
            protocol_path = root / "protocol_manifest.json"
            protocol_path.write_text(json.dumps(protocol_manifest))
            main([
                "--run",
                str(run),
                "--out",
                str(root / "out"),
                "--manifest-json",
                str(protocol_path),
                "--sensitivity-json",
                str(protocol_path),
                "--bootstrap-draws",
                "200",
            ])
            summary = json.loads((root / "out" / "summary.json").read_text())
            self.assertEqual(summary["sensitivity"]["mode"], "sensitivity_pair_indices")
            self.assertEqual(summary["sensitivity"]["n_pairs"], 101)
            self.assertEqual(summary["sensitivity"]["removed_pairs"], 27)
            self.assertIn("not certified clean", summary["sensitivity"]["interpretation"])
            self.assertEqual(summary["pose_qualification"]["body"]["label"], "joint_detection_support")
            report = (root / "out" / "REPORT.md").read_text()
            self.assertIn("joint_detection_support", report)
            self.assertIn("真实资格来自 M scores>=.3", report)


if __name__ == "__main__":
    unittest.main()
