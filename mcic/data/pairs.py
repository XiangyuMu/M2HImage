from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class ImagePair:
    sample_id: str
    mannequin_path: Path
    human_path: Path

    def json_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


def _index_images(folder: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    if not folder.exists():
        return paths
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            relative_id = str(path.relative_to(folder).with_suffix(""))
            if relative_id in paths:
                raise ValueError(f"Duplicate image stem in {folder}: {relative_id}")
            paths[relative_id] = path
    return paths


def discover_pairs(root: Path, mannequin_dir: str, human_dir: str) -> tuple[list[ImagePair], dict[str, list[str]]]:
    mannequin = _index_images(root / mannequin_dir)
    human = _index_images(root / human_dir)
    shared = sorted(set(mannequin) & set(human))
    pairs = [ImagePair(key, mannequin[key], human[key]) for key in shared]
    failures = {
        "missing_human": sorted(set(mannequin) - set(human)),
        "missing_mannequin": sorted(set(human) - set(mannequin)),
    }
    return pairs, failures
