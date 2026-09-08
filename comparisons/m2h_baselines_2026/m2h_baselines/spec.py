from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SampleSpec:
    """One fully resolved paired or counterfactual M2H sample."""

    key: str
    split: str
    mid: str
    jid: str
    target: str
    mannequin: str
    agnostic: str
    replace_mask: str
    identity_card: str
    face: str
    face_region: str
    garment: str
    pose: str
    resolution: str
    content_box: tuple[int, int, int, int]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["content_box"] = list(self.content_box)
        return payload

    def resolve(self, root: Path, field: str) -> Path:
        return root / str(getattr(self, field))


IMAGE_FIELDS = (
    "target",
    "mannequin",
    "agnostic",
    "replace_mask",
    "identity_card",
    "face",
    "face_region",
    "garment",
    "pose",
)

