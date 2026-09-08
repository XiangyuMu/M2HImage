from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


class BlockProtocolError(ValueError):
    pass


CELL_ORDER = ((0, 0), (0, 1), (1, 0), (1, 1))
REQUIRED_CELL_FIELDS = {
    "block_id",
    "cell_id",
    "identity_index",
    "mannequin_index",
    "identity_id",
    "mannequin_id",
    "noise_seed",
    "noise_hash",
    "split",
    "source_dataset",
    "intervention_type",
    "garment_sku",
    "context_id",
    "input_identity_path",
    "input_mannequin_path",
}


@dataclass(frozen=True)
class BlockCell:
    block_id: str
    cell_id: str
    identity_index: int
    mannequin_index: int
    identity_id: str
    mannequin_id: str
    noise_seed: int
    noise_hash: str
    split: str
    source_dataset: str
    intervention_type: str
    garment_sku: str
    context_id: str
    input_identity_path: str
    input_mannequin_path: str


def _text(row: Mapping[str, Any], key: str) -> str:
    value = str(row.get(key, "")).strip()
    if not value:
        raise BlockProtocolError(f"missing {key} in block cell")
    return value


def _integer(row: Mapping[str, Any], key: str) -> int:
    try:
        return int(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise BlockProtocolError(f"invalid integer {key}") from exc


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def manifest_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    normalized = [dict(sorted((str(k), row[k]) for k in row)) for row in rows]
    return hashlib.sha256(_canonical_json(normalized)).hexdigest()


def validate_cell_row(row: Mapping[str, Any]) -> BlockCell:
    missing = REQUIRED_CELL_FIELDS - set(row)
    if missing:
        raise BlockProtocolError(f"missing fields: {sorted(missing)}")
    identity_index = _integer(row, "identity_index")
    mannequin_index = _integer(row, "mannequin_index")
    if identity_index not in (0, 1) or mannequin_index not in (0, 1):
        raise BlockProtocolError("identity_index and mannequin_index must be 0 or 1")
    cell_id = _text(row, "cell_id")
    expected_cell = f"I{identity_index}M{mannequin_index}"
    if cell_id != expected_cell:
        raise BlockProtocolError(f"cell_id {cell_id!r} does not match {expected_cell!r}")
    noise_hash = _text(row, "noise_hash")
    if len(noise_hash) != 64 or any(char not in "0123456789abcdefABCDEF" for char in noise_hash):
        raise BlockProtocolError("noise_hash must be a SHA-256 hex digest")
    return BlockCell(
        block_id=_text(row, "block_id"),
        cell_id=cell_id,
        identity_index=identity_index,
        mannequin_index=mannequin_index,
        identity_id=_text(row, "identity_id"),
        mannequin_id=_text(row, "mannequin_id"),
        noise_seed=_integer(row, "noise_seed"),
        noise_hash=noise_hash.lower(),
        split=_text(row, "split"),
        source_dataset=_text(row, "source_dataset"),
        intervention_type=_text(row, "intervention_type"),
        garment_sku=_text(row, "garment_sku"),
        context_id=_text(row, "context_id"),
        input_identity_path=_text(row, "input_identity_path"),
        input_mannequin_path=_text(row, "input_mannequin_path"),
    )


def _same_across(cells: list[BlockCell], field: str) -> bool:
    return len({getattr(cell, field) for cell in cells}) == 1


def validate_complete_blocks(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_split: str | None = None,
    forbidden_identity_ids: set[str] | None = None,
    forbidden_image_paths: set[str] | None = None,
    forbidden_garment_skus: set[str] | None = None,
    forbidden_context_ids: set[str] | None = None,
) -> dict[str, Any]:
    parsed = [validate_cell_row(row) for row in rows]
    if not parsed:
        raise BlockProtocolError("manifest is empty")
    by_block: dict[str, list[BlockCell]] = {}
    for cell in parsed:
        by_block.setdefault(cell.block_id, []).append(cell)
    errors: list[str] = []
    block_records = []
    for block_id, cells in sorted(by_block.items()):
        coordinates = {(cell.identity_index, cell.mannequin_index) for cell in cells}
        if len(cells) != 4 or coordinates != set(CELL_ORDER):
            errors.append(f"{block_id}: expected exactly one of each I0M0/I0M1/I1M0/I1M1")
            continue
        if not _same_across(cells, "noise_seed") or not _same_across(cells, "noise_hash"):
            errors.append(f"{block_id}: noise is not identical across all four cells")
        for field in ("split", "source_dataset"):
            if not _same_across(cells, field):
                errors.append(f"{block_id}: {field} must be fixed within a block")
        intervention_type = cells[0].intervention_type
        if intervention_type not in {"garment", "context"}:
            errors.append(f"{block_id}: intervention_type must be garment or context")
        cell_by_coordinate = {(item.identity_index, item.mannequin_index): item for item in cells}
        for j in (0, 1):
            if cell_by_coordinate[(0, j)].identity_id == cell_by_coordinate[(1, j)].identity_id:
                errors.append(f"{block_id}: identity swap did not change identity_id")
            if cell_by_coordinate[(0, j)].input_mannequin_path != cell_by_coordinate[(1, j)].input_mannequin_path:
                errors.append(f"{block_id}: identity swap changed mannequin input")
        for i in (0, 1):
            if cell_by_coordinate[(i, 0)].mannequin_id == cell_by_coordinate[(i, 1)].mannequin_id:
                errors.append(f"{block_id}: mannequin swap did not change mannequin_id")
            if cell_by_coordinate[(i, 0)].input_identity_path != cell_by_coordinate[(i, 1)].input_identity_path:
                errors.append(f"{block_id}: mannequin swap changed identity input")
            if intervention_type == "garment":
                if cell_by_coordinate[(i, 0)].context_id != cell_by_coordinate[(i, 1)].context_id:
                    errors.append(f"{block_id}: garment intervention changed context_id")
                if cell_by_coordinate[(i, 0)].garment_sku == cell_by_coordinate[(i, 1)].garment_sku:
                    errors.append(f"{block_id}: garment intervention did not change garment_sku")
            elif intervention_type == "context":
                if cell_by_coordinate[(i, 0)].garment_sku != cell_by_coordinate[(i, 1)].garment_sku:
                    errors.append(f"{block_id}: context intervention changed garment_sku")
                if cell_by_coordinate[(i, 0)].context_id == cell_by_coordinate[(i, 1)].context_id:
                    errors.append(f"{block_id}: context intervention did not change context_id")
        if expected_split is not None and any(cell.split != expected_split for cell in cells):
            errors.append(f"{block_id}: unexpected split")
        for cell in cells:
            if forbidden_identity_ids and cell.identity_id in forbidden_identity_ids:
                errors.append(f"{block_id}: identity overlap {cell.identity_id}")
            if forbidden_garment_skus and cell.garment_sku in forbidden_garment_skus:
                errors.append(f"{block_id}: garment SKU overlap {cell.garment_sku}")
            if forbidden_context_ids and cell.context_id in forbidden_context_ids:
                errors.append(f"{block_id}: context overlap {cell.context_id}")
            if forbidden_image_paths and ({cell.input_identity_path, cell.input_mannequin_path} & forbidden_image_paths):
                errors.append(f"{block_id}: image overlap")
        block_records.append({
            "block_id": block_id,
            "cells": [cell.cell_id for cell in sorted(cells, key=lambda item: item.cell_id)],
            "noise_seed": cell_by_coordinate[(0, 0)].noise_seed,
            "noise_hash": cell_by_coordinate[(0, 0)].noise_hash,
        })
    if errors:
        raise BlockProtocolError("; ".join(errors))
    return {
        "block_count": len(by_block),
        "cell_count": len(parsed),
        "complete": True,
        "manifest_sha256": manifest_sha256([cell.__dict__ for cell in parsed]),
        "blocks": block_records,
    }


def deterministic_replay(rows: Iterable[Mapping[str, Any]], replay_rows: Iterable[Mapping[str, Any]]) -> bool:
    left = [dict(sorted((str(k), row[k]) for k in row)) for row in rows]
    right = [dict(sorted((str(k), row[k]) for k in row)) for row in replay_rows]
    return _canonical_json(left) == _canonical_json(right)


def intention_to_test_output_status(
    manifest_rows: Iterable[Mapping[str, Any]],
    output_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Count missing/invalid outputs without dropping their intended cells."""
    expected = {(str(row["block_id"]), str(row["cell_id"])) for row in manifest_rows}
    observed = set()
    valid = set()
    for row in output_rows:
        key = (str(row.get("block_id", "")), str(row.get("cell_id", "")))
        observed.add(key)
        if bool(row.get("valid_output", False)):
            valid.add(key)
    unexpected = observed - expected
    missing = expected - observed
    invalid = (observed & expected) - valid
    if unexpected:
        raise BlockProtocolError(f"outputs contain unexpected cells: {sorted(unexpected)}")
    return {
        "intention_to_test_cells": len(expected),
        "valid_cells": len(valid),
        "missing_cells": len(missing),
        "invalid_cells": len(invalid),
        "worst_case_cells": len(missing | invalid),
        "valid_output_rate": (len(valid) / len(expected)) if expected else 0.0,
    }
