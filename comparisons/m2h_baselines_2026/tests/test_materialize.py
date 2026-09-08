from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from m2h_baselines.materialize import materialize


class MaterializeTest(unittest.TestCase):
    def test_all_layouts_keep_source_and_target_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prepared = root / "prepared"
            fields = ("target", "mannequin", "agnostic", "replace_mask", "identity_card", "face", "face_region", "garment", "pose")
            record = {"key": "train:00001:00001", "split": "train", "mid": "00001", "jid": "00001", "resolution": "low", "content_box": [64, 0, 448, 512]}
            for field in fields:
                path = Path("low/train") / field / "00001.png"
                destination = prepared / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                mode = "L" if field in {"replace_mask", "face_region"} else "RGB"
                Image.new(mode, (512, 512), 0 if mode == "L" else (20, 40, 60)).save(destination)
                record[field] = str(path)
            identity_person = prepared / "low/train/identity_person/00001.png"
            identity_person.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (512, 512), (60, 80, 100)).save(identity_person)
            pose_dense = prepared / "low/train/pose_dense/00001.png"
            pose_dense.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (512, 512), (80, 100, 120)).save(pose_dense)
            replace = Image.open(prepared / record["replace_mask"])
            replace.paste(255, (250, 100, 270, 140))
            replace.save(prepared / record["replace_mask"])
            face_region = Image.open(prepared / record["face_region"])
            face_region.paste(255, (240, 80, 280, 120))
            face_region.save(prepared / record["face_region"])
            manifest = root / "samples.jsonl"
            manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
            layouts = root / "layouts"
            for method in ("refton", "ita_mdt", "idm_vton", "mcld"):
                materialize(method, manifest, prepared, layouts)
            self.assertTrue((layouts / "refton/train/person/00001.png").is_symlink())
            self.assertTrue((layouts / "refton/train/image/00001.png").is_symlink())
            self.assertNotEqual(
                (layouts / "refton/train/person/00001.png").resolve(),
                (layouts / "refton/train/image/00001.png").resolve(),
            )
            self.assertEqual(
                (layouts / "refton/train/person/00001.png").resolve(),
                (prepared / record["mannequin"]).resolve(),
            )
            self.assertEqual(
                (layouts / "refton/train/cloth/00001.png").resolve(),
                (prepared / record["face"]).resolve(),
            )
            self.assertEqual(
                (layouts / "refton/train/image_ref/00001.png").resolve(),
                (prepared / record["identity_card"]).resolve(),
            )
            self.assertTrue((layouts / "idm_vton/train/person/00001.png").is_symlink())
            self.assertEqual(
                (layouts / "idm_vton/train/person/00001.png").resolve(),
                (prepared / record["mannequin"]).resolve(),
            )
            ita_mask = layouts / "ita_mdt/zalando-hd-resized/train/agnostic-mask/00001_mask.png"
            self.assertFalse(ita_mask.is_symlink())
            ita_values = Image.open(ita_mask).convert("L")
            self.assertEqual(ita_values.getpixel((208, 48)), 255)
            self.assertEqual(ita_values.getpixel((207, 47)), 0)
            self.assertEqual(ita_values.getpixel((255, 110)), 255)
            self.assertEqual(
                (layouts / "ita_mdt/zalando-hd-resized/train/cloth/00001.png").resolve(),
                identity_person.resolve(),
            )
            self.assertEqual(
                (
                    layouts
                    / "ita_mdt/zalando-hd-resized/train/image-densepose/00001.png"
                ).resolve(),
                pose_dense.resolve(),
            )
            self.assertTrue((layouts / "mcld/train.csv").exists())
            self.assertEqual(
                (layouts / "mcld/train_texture/00001_identity.png").resolve(),
                (prepared / record["garment"]).resolve(),
            )
            self.assertEqual(
                (layouts / "mcld/train_face_inputs/00001_identity.png").resolve(),
                (prepared / record["face"]).resolve(),
            )


if __name__ == "__main__":
    unittest.main()
