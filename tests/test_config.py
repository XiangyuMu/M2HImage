from pathlib import Path

import pytest
import yaml

from mcic.utils.config import load_config


def write_config(path: Path):
    payload = {
        "data": {"root": "/tmp/data", "train_ratio": 0.8, "val_ratio": 0.1},
        "image": {"height": 512, "width": 512},
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_resolution_override(tmp_path):
    path = tmp_path / "config.yaml"
    write_config(path)
    config = load_config(path, ["image.height=768", "image.width=512"])
    assert config["image"]["height"] == 768


def test_resolution_must_be_multiple_of_64(tmp_path):
    path = tmp_path / "config.yaml"
    write_config(path)
    with pytest.raises(ValueError, match="multiple of 64"):
        load_config(path, ["image.width=500"])
