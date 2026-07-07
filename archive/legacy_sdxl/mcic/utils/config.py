from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any, Iterable

import yaml


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config = copy.deepcopy(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must use key=value syntax: {override}")
        dotted_key, raw_value = override.split("=", 1)
        _set_nested(config, dotted_key, yaml.safe_load(raw_value))
    validate_config(config)
    return config


def _set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = config
    parts = dotted_key.split(".")
    for key in parts[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[parts[-1]] = value


def validate_config(config: dict[str, Any]) -> None:
    image = config.get("image", {})
    for key in ("height", "width"):
        value = image.get(key)
        if not isinstance(value, int) or value <= 0 or value % 64:
            raise ValueError(f"image.{key} must be a positive multiple of 64, received {value!r}")
    ratios = config.get("data", {})
    train_ratio = float(ratios.get("train_ratio", 0.8))
    val_ratio = float(ratios.get("val_ratio", 0.1))
    if train_ratio < 0 or val_ratio < 0 or train_ratio + val_ratio > 1:
        raise ValueError("data.train_ratio and data.val_ratio must define valid dataset splits")


def add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="YAML configuration path.")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional dotted YAML overrides, for example image.height=768.",
    )


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    value = Path(value)
    if value.is_absolute():
        return value
    root = Path(config.get("data", {}).get("root", "."))
    return root / value


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def seeded_split(sample_ids: list[str], config: dict[str, Any]) -> dict[str, str]:
    ids = list(sample_ids)
    random.Random(config["data"].get("split_seed", 42)).shuffle(ids)
    n_train = int(len(ids) * float(config["data"].get("train_ratio", 0.8)))
    n_val = int(len(ids) * float(config["data"].get("val_ratio", 0.1)))
    result: dict[str, str] = {}
    for position, sample_id in enumerate(ids):
        result[sample_id] = (
            "train" if position < n_train else "val" if position < n_train + n_val else "test"
        )
    return result
