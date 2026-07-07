import json

from PIL import Image

from mcic.preprocess.run import preprocess_dataset


def test_preprocessing_creates_metadata_and_masks(tmp_path):
    for folder in ("mannequin", "human"):
        (tmp_path / folder).mkdir()
    for index in range(3):
        image = Image.new("RGB", (128, 192), (180 + index, 120, 100))
        image.save(tmp_path / "mannequin" / f"{index:04d}.jpg")
        image.save(tmp_path / "human" / f"{index:04d}.jpg")
    config = {
        "data": {
            "root": str(tmp_path),
            "mannequin_dir": "mannequin",
            "human_dir": "human",
            "cache_dir": "cache_mcic",
            "split_seed": 42,
            "train_ratio": 0.8,
            "val_ratio": 0.1,
        },
        "image": {"height": 512, "width": 512},
        "preprocess": {
            "parsing_backend": "heuristic",
            "boundary_dilate_radius": 3,
            "cloth_erode_radius": 1,
            "max_visual_checks": 2,
        },
        "identity": {"backend": "mock", "embedding_dim": 512, "min_face_size": 10},
    }
    path = preprocess_dataset(config)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    assert rows[0]["face_quality_pass"]
    assert (tmp_path / "cache_mcic" / "audit_report.json").exists()
