import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import region_diagnostics as rd


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def base_row(mid: str, jid: str, branch: str = "a2_inputOnly_freshI", seed: int = 0) -> dict:
    return {
        "mid": mid,
        "jid": jid,
        "branch": branch,
        "tau": 0,
        "k": 1,
        "seed": seed,
        "path": f"/tmp/{mid}_{jid}_{branch}_{seed}.png",
    }


def full_grid(mids: tuple[str, ...] = ("00430",)) -> list[dict]:
    rows = []
    for mid in mids:
        for branch in rd.EXPECTED_BRANCHES:
            rows.append(base_row(mid, "28234", branch))
            rows.append(base_row(mid, "31226", branch))
    return rows


class RegionDiagnosticsTests(unittest.TestCase):
    def test_partition_source_mask_fixed_regions(self) -> None:
        source = np.zeros((9, 9, 3), dtype=np.uint8)
        source[:, 5:, :] = 255
        mask = np.zeros((9, 9), dtype=bool)
        mask[1:8, 1:8] = True

        regions, qualification = rd.partition_source_mask(source, mask)

        self.assertEqual(set(regions), {"garment", "interior", "boundary", "hightexture", "lowtexture"})
        self.assertEqual(qualification["erode_radius_px"], 1)
        self.assertEqual(regions["garment"].sum(), 49)
        self.assertEqual(regions["interior"].sum(), 25)
        self.assertEqual(regions["boundary"].sum(), 24)
        self.assertTrue(np.array_equal(regions["boundary"], regions["garment"] & ~regions["interior"]))
        self.assertTrue(np.array_equal(regions["hightexture"] | regions["lowtexture"], regions["garment"]))
        self.assertFalse(np.any(regions["hightexture"] & regions["lowtexture"]))
        self.assertEqual(qualification["region_support"]["garment"], 49)
        self.assertIs(qualification["region_eligible"]["interior"], True)

    def test_partition_empty_mask_records_empty_regions(self) -> None:
        source = np.zeros((6, 6, 3), dtype=np.uint8)
        mask = np.zeros((6, 6), dtype=bool)

        regions, qualification = rd.partition_source_mask(source, mask)

        self.assertTrue(all(region.sum() == 0 for region in regions.values()))
        self.assertIsNone(qualification["texture_threshold"])
        self.assertEqual(
            qualification["region_eligible"],
            {
                "garment": False,
                "interior": False,
                "boundary": False,
                "hightexture": False,
                "lowtexture": False,
            },
        )

    def test_flat_source_gradient_does_not_make_entire_mask_hightexture(self) -> None:
        source = np.zeros((6, 6, 3), dtype=np.uint8)
        mask = np.ones((6, 6), dtype=bool)

        regions, qualification = rd.partition_source_mask(source, mask)

        self.assertEqual(regions["hightexture"].sum(), 0)
        self.assertFalse(qualification["region_eligible"]["hightexture"])
        self.assertEqual(qualification["positive_gradient_pixels"], 0)

    def test_erode_uses_zero_padding_at_image_edge(self) -> None:
        mask = np.ones((5, 5), dtype=bool)

        eroded = rd.binary_erode(mask, radius=1)

        expected = np.zeros((5, 5), dtype=bool)
        expected[1:4, 1:4] = True
        self.assertTrue(np.array_equal(eroded, expected))

    def test_load_rows_rejects_duplicate_row_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rows_path = Path(tmp) / "rows.jsonl"
            rows = full_grid()
            rows.append(dict(rows[0]))
            write_rows(rows_path, rows)

            with self.assertRaisesRegex(ValueError, "duplicate diagnostic rows"):
                rd.load_rows(rows_path)

    def test_load_rows_rejects_missing_two_reference_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rows_path = Path(tmp) / "rows.jsonl"
            rows = full_grid()
            rows = [row for row in rows if not (row["branch"] == "b2_inputOnly_freshI" and row["jid"] == "31226")]
            write_rows(rows_path, rows)

            with self.assertRaisesRegex(ValueError, "missing or invalid 2-reference pair keys"):
                rd.load_rows(rows_path)

    def test_load_rows_rejects_missing_complete_arm_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rows_path = Path(tmp) / "rows.jsonl"
            rows = [row for row in full_grid(("00430", "00431")) if not (row["branch"] == "a4_inputOnly_freshI" and row["mid"] == "00431")]
            write_rows(rows_path, rows)

            with self.assertRaisesRegex(ValueError, "all observed arm pair sets must be identical"):
                rd.load_rows(rows_path)

    def test_load_rows_rejects_unexpected_tau_k_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rows_path = Path(tmp) / "rows.jsonl"
            rows = full_grid()
            rows[0]["seed"] = 1
            write_rows(rows_path, rows)

            with self.assertRaisesRegex(ValueError, "expected only tau0/k1/seed0 rows"):
                rd.load_rows(rows_path)

    def test_load_rows_accepts_exact_full_arm_grid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rows_path = Path(tmp) / "rows.jsonl"
            rows = full_grid()
            write_rows(rows_path, rows)

            loaded = rd.load_rows(rows_path)

            self.assertEqual(len(loaded), 6)
            self.assertEqual({row["branch"] for row in loaded}, set(rd.EXPECTED_BRANCHES))


if __name__ == "__main__":
    unittest.main()
