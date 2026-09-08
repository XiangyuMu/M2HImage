from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from m2h_baselines.prepare import prepare_split
from m2h_baselines.validate import validate_manifest


class PrepareTest(unittest.TestCase):
    def test_low_resolution_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "dataset"
            out = Path(temp) / "prepared"
            sample_id = "00001"
            directories = [
                "images/human",
                "images/mannequin",
                "derived/face_crops/human",
                "derived/region_masks",
                "derived/head_pose_6drepnet/human",
                "human_parsing/fashn/masks/mannequin",
                "dwpose/without_head/mannequin",
                "dwpose/with_head/mannequin",
                "dwpose/keypoints/mannequin",
                "splits",
            ]
            for directory in directories:
                (root / directory).mkdir(parents=True, exist_ok=True)
            rgb = np.full((1024, 768, 3), 220, dtype=np.uint8)
            rgb[200:900, 180:590] = (50, 100, 150)
            Image.fromarray(rgb).save(root / "images/human" / f"{sample_id}.png")
            Image.fromarray(rgb[:, ::-1].copy()).save(root / "images/mannequin" / f"{sample_id}.png")
            Image.fromarray(rgb[120:360, 260:500]).save(root / "derived/face_crops/human" / f"{sample_id}.png")
            parsing = np.zeros((1024, 768), dtype=np.uint8)
            parsing[100:300, 260:510] = 1
            parsing[300:800, 180:590] = 3
            parsing[800:950, 240:540] = 14
            Image.fromarray(parsing).save(root / "human_parsing/fashn/masks/mannequin" / f"{sample_id}.png")
            cloth = np.zeros((1024, 768), dtype=np.uint8)
            cloth[300:800, 180:590] = 255
            identity = np.zeros_like(cloth)
            identity[100:300, 260:510] = 255
            np.savez(
                root / "derived/region_masks" / f"{sample_id}.npz",
                cloth=cloth,
                cloth_safe=cloth,
                id_strong=identity,
                id_weak=np.zeros_like(cloth),
                edge=np.zeros_like(cloth),
                body_bg=np.zeros_like(cloth),
            )
            Image.new("RGB", (768, 1024), (0, 0, 0)).save(
                root / "dwpose/without_head/mannequin" / f"{sample_id}.png"
            )
            Image.new("RGB", (768, 1024), (0, 0, 0)).save(
                root / "dwpose/with_head/mannequin" / f"{sample_id}.png"
            )
            body = np.zeros((18, 2), dtype=np.float32)
            body[1] = (0.5, 0.32)
            body[2], body[5] = (0.35, 0.38), (0.65, 0.38)
            np.savez(
                root / "dwpose/keypoints/mannequin" / f"{sample_id}.npz",
                body=body,
                body_scores=np.ones(18, dtype=np.float32),
            )
            (root / "derived/head_pose_6drepnet/human" / f"{sample_id}.json").write_text(
                json.dumps({"status": "ok", "yaw": 10, "pitch": 2, "roll": -3}), encoding="utf-8"
            )
            (root / "splits/train.txt").write_text(sample_id + "\n", encoding="utf-8")
            manifest = prepare_split(
                data_root=root,
                output_root=out,
                split="train",
                resolution="low",
                prep_config={
                    "neutral_rgb": [127, 127, 127],
                    "replace_labels": [1, 2, 12, 13, 14, 15, 16],
                    "garment_labels": [3, 4, 5, 6, 7],
                    "mask_dilation": 9,
                    "cloth_protect_dilation": 5,
                    "identity_crop_padding": 0.22,
                    "identity_person_crop_padding": 0.12,
                    "garment_crop_padding": 0.12,
                    "head_nose_offset_scale": 1.62,
                },
                workers=1,
            )
            report = validate_manifest(manifest, out)
            self.assertEqual(report["status"], "pass")
            record = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(record["content_box"], [64, 0, 448, 512])
            identity_person = out / "low/train/identity_person/00001.png"
            self.assertTrue(identity_person.is_file())
            with Image.open(identity_person) as image:
                self.assertEqual(image.size, (512, 512))
            pose_dense = out / "low/train/pose_dense/00001.png"
            self.assertTrue(pose_dense.is_file())
            with Image.open(pose_dense) as image:
                self.assertEqual(image.size, (512, 512))


if __name__ == "__main__":
    unittest.main()
