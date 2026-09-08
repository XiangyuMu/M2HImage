import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from protocol_revalidation import collect_near_candidates, exact_human_train_overlaps, load_pairs, main


def write_image(path, color):
    Image.new("RGB", (24, 32), color).save(path)


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class ProtocolRevalidationTests(unittest.TestCase):
    def test_load_pairs_preserves_128_contract(self):
        pairs = []
        for i in range(64):
            mid = f"m{i:03d}"
            pairs.append({"mid": mid, "jid": f"j{i:03d}a", "seed": 0})
            pairs.append({"mid": mid, "jid": f"j{i:03d}b", "seed": 0})
        payload = {"pairs": pairs, "schema_version": "old"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dev128_pairs.json"
            path.write_text(json.dumps(payload))
            self.assertEqual(load_pairs(path), payload)
            self.assertEqual(json.loads(path.read_text()), payload)

    def test_exact_human_train_overlap_list(self):
        per_image = {
            ("dev1", "human"): {"path": "/dev1.jpg"},
            ("train1", "human"): {"path": "/train1.jpg"},
        }
        groups = [{"role": "human", "sha256": "a" * 64,
                   "members": [{"id": "dev1", "split": "val"}, {"id": "train1", "split": "train"}]}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "exact_groups.json"
            path.write_text(json.dumps(groups))
            rows = exact_human_train_overlaps(path, per_image)
            self.assertEqual(rows[0]["id"], "dev1")
            self.assertEqual(rows[0]["train_ids"], "train1")

    def test_collect_near_candidates_filters_dev_human_vs_train_human(self):
        images = [
            {"id": "dev", "role": "human", "split": "val", "path": "d"},
            {"id": "tr", "role": "human", "split": "train", "path": "t"},
            {"id": "trm", "role": "mannequin", "split": "train", "path": "m"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image_pairs.csv.gz"
            with gzip.open(path, "wt", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["left_image_index", "right_image_index", "hamming", "sha256_equal"])
                writer.writeheader()
                writer.writerow({"left_image_index": 0, "right_image_index": 1, "hamming": 2, "sha256_equal": 0})
                writer.writerow({"left_image_index": 0, "right_image_index": 2, "hamming": 2, "sha256_equal": 0})
            rows = collect_near_candidates(path, images, {"dev"})
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["train_id"], "tr")

    def test_end_to_end_fixture_and_fresh_out_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            old_protocol = base / "old"
            artifacts = base / "art"
            full_dup = artifacts / "full_duplicates"
            group = artifacts / "protocol_group_audit_v1" / "results"
            near = artifacts / "protocol_near_duplicate_v1" / "results"
            for d in (old_protocol, full_dup, group, near, base / "img"):
                d.mkdir(parents=True)
            ids = [f"m{i:03d}" for i in range(64)] + [f"j{i:03d}a" for i in range(64)] + [f"j{i:03d}b" for i in range(64)] + [f"t{i:03d}" for i in range(33)]
            paths = {}
            for n, sid in enumerate(ids):
                path = base / "img" / f"{sid}.png"
                write_image(path, ((n * 31) % 255, (n * 17) % 255, (n * 7) % 255))
                paths[sid] = path
            pairs = []
            for i in range(64):
                pairs.append({"mid": f"m{i:03d}", "jid": f"j{i:03d}a", "seed": 0})
                pairs.append({"mid": f"m{i:03d}", "jid": f"j{i:03d}b", "seed": 0})
            (old_protocol / "dev128_pairs.json").write_text(json.dumps({"pairs": pairs}))
            per_rows = []
            for sid in ids:
                split = "train" if sid.startswith("t") else "val"
                for role in ("human", "mannequin"):
                    per_rows.append({"id": sid, "role": role, "split": split, "status": "ok",
                                     "path": str(paths[sid]), "sha256": f"{sid}{role}".encode().hex().ljust(64, "0")[:64],
                                     "dhash": "0" * 16})
            with (full_dup / "per_image.jsonl").open("w") as f:
                for row in per_rows:
                    f.write(json.dumps(row) + "\n")
            (full_dup / "exact_groups.json").write_text(json.dumps([]))
            candidate_rows = []
            for sid in ids:
                candidate_rows.append({"id": sid, "human_path": str(paths[sid]), "mannequin_path": str(paths[sid]),
                                       "component_id": "c_" + sid, "source_group": "sg_" + sid})
            write_csv(group / "candidate_all.csv", candidate_rows)
            with gzip.open(near / "image_pairs.csv.gz", "wt", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["left_image_index", "right_image_index", "hamming", "sha256_equal"])
                writer.writeheader()
                writer.writerow({"left_image_index": 0, "right_image_index": len(ids) * 2 - 2, "hamming": 3, "sha256_equal": 0})
            out = base / "out"
            import sys
            old_argv = sys.argv
            try:
                sys.argv = ["protocol_revalidation.py", "--old-protocol", str(old_protocol), "--artifacts", str(artifacts), "--out", str(out)]
                main()
                self.assertTrue((out / "PROTOCOL_REVALIDATED").is_file())
                summary = json.loads((out / "summary.json").read_text())
                self.assertEqual(summary["pair_count"], 128)
                self.assertEqual(summary["dev128_exact_train_human_hits"], 0)
                with self.assertRaises(SystemExit):
                    main()
            finally:
                sys.argv = old_argv


if __name__ == "__main__":
    unittest.main()
