from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from tools.build_role_flow_eval_manifest import build_manifest
from tools.generate_role_flow_eval import (
    build_generation_inputs,
    canonical_sha256,
    load_manifest,
    select_rows,
    validate_or_prepare_output,
)


def test_build_manifest_is_deterministic_and_excludes_quarantine(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    human_a = source / "a.jpg"
    human_b = source / "b.jpg"
    mannequin_a = source / "a.png"
    mannequin_b = source / "b.png"
    human_c = source / "c.jpg"
    mannequin_c = source / "c.png"
    human_a.write_bytes(b"human-a")
    human_b.write_bytes(b"human-b")
    mannequin_a.write_bytes(b"mannequin-a")
    mannequin_b.write_bytes(b"mannequin-b")
    human_c.write_bytes(b"human-c")
    mannequin_c.write_bytes(b"mannequin-c")
    report = tmp_path / "report"
    report.mkdir()
    person = report / "person.csv"
    with person.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "source_dataset", "source_key", "human_path", "mannequin_path", "clean_split", "person_cluster_id"])
        writer.writeheader()
        writer.writerows([
            {"id": "a", "source_dataset": "x", "source_key": "ka", "human_path": str(human_a), "mannequin_path": str(mannequin_a), "clean_split": "test", "person_cluster_id": "ca"},
            {"id": "b", "source_dataset": "x", "source_key": "kb", "human_path": str(human_b), "mannequin_path": str(mannequin_b), "clean_split": "quarantine", "person_cluster_id": "cb"},
            {"id": "c", "source_dataset": "x", "source_key": "kc", "human_path": str(human_c), "mannequin_path": str(mannequin_c), "clean_split": "val", "person_cluster_id": "cc"},
        ])
    split = tmp_path / "splits"
    split.mkdir()
    (split / "train.txt").write_text("", encoding="utf-8")
    (split / "val.txt").write_text("c\n", encoding="utf-8")
    (split / "test.txt").write_text("a\n", encoding="utf-8")
    (split / "quarantine.txt").write_text("b\n", encoding="utf-8")
    pair = tmp_path / "pairs.json"
    pair.write_text(json.dumps({"pairs": [{"mannequin_id": "a", "identity_id": "a", "seeds": [3]}]}), encoding="utf-8")
    val_pair = tmp_path / "val_pairs.json"
    val_pair.write_text(json.dumps({"pairs": [{"mannequin_id": "c", "identity_id": "c", "seeds": [4]}]}), encoding="utf-8")
    output = tmp_path / "manifest.json"
    payload = build_manifest(dataset_root=tmp_path, person_manifest_path=person, split_dir=split, output=output, pair_source=pair, val_pair_source=val_pair)
    assert payload["content_sha256"]
    assert all(row["mid"] != "b" and row["jid"] != "b" for row in payload["rows"])
    loaded = load_manifest(output)
    assert select_rows(loaded, "test", None)[0]["mannequin_path"] == str(mannequin_a.resolve())


def _write_png(path: Path, size: tuple[int, int] = (512, 512), mode: str = "RGB") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    color = (11, 22, 33) if mode == "RGB" else 11
    Image.new(mode, size, color=color).save(path)


def _generation_inputs(
    tmp_path: Path,
    *,
    generate_steps: int = 4,
    generation_git_commit: str | None = None,
    generation_script_sha256: str | None = None,
) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "checkpoint"
    manifest = tmp_path / "manifest.json"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    config.write_text("eval:\n  generate_steps: 4\n", encoding="utf-8")
    payload = {
        "schema_version": "m2h-role-flow-eval-v1",
        "content_sha256": "manifest-content",
        "rows": [
            {
                "split": "val",
                "mid": "m1",
                "jid": "j1",
                "seed": 0,
                "mannequin_path": "/tmp/m1.png",
                "human_path": "/tmp/j1.png",
            },
            {
                "split": "val",
                "mid": "m2",
                "jid": "j2",
                "seed": 1,
                "mannequin_path": "/tmp/m2.png",
                "human_path": "/tmp/j2.png",
            },
        ],
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    cfg = {"eval": {"generate_steps": generate_steps}, "data": {"resolution": 512}}
    return build_generation_inputs(
        cfg=cfg,
        config=config,
        checkpoint=checkpoint,
        manifest=manifest,
        manifest_payload=payload,
        rows=payload["rows"],
        split="val",
        limit=None,
        generation_git_commit=generation_git_commit,
        generation_script_sha256=generation_script_sha256,
    )


def test_finalize_only_writes_complete_generation_provenance(tmp_path: Path) -> None:
    inputs = _generation_inputs(tmp_path)
    output = tmp_path / "outputs"
    for name in inputs["selection"]["expected_filenames"]:
        _write_png(output / name)
    payload = validate_or_prepare_output(output, inputs, finalize_only=True, device="cpu")
    assert payload["schema_version"] == "m2h-role-flow-generation-v2"
    assert payload["status"] == "complete"
    assert payload["selection"]["selected_row_count"] == 2
    assert payload["selection"]["expected_filenames"] == ["m1__idj1__seed0.png", "m2__idj2__seed1.png"]
    assert len(payload["images"]) == 2
    assert {record["readable"] for record in payload["images"]} == {True}
    assert (output / "generation_provenance.json").is_file()


def test_finalize_only_rejects_extra_png(tmp_path: Path) -> None:
    inputs = _generation_inputs(tmp_path)
    output = tmp_path / "outputs"
    for name in inputs["selection"]["expected_filenames"]:
        _write_png(output / name)
    _write_png(output / "extra.png")
    with pytest.raises(ValueError, match="extra_png"):
        validate_or_prepare_output(output, inputs, finalize_only=True, device="cpu")
    payload = json.loads((output / "generation_provenance.json").read_text(encoding="utf-8"))
    assert payload["status"] == "failed"


def test_finalize_only_rejects_non_rgb_png(tmp_path: Path) -> None:
    inputs = _generation_inputs(tmp_path)
    output = tmp_path / "outputs"
    _write_png(output / inputs["selection"]["expected_filenames"][0], mode="L")
    _write_png(output / inputs["selection"]["expected_filenames"][1])
    with pytest.raises(ValueError, match="non_rgb"):
        validate_or_prepare_output(output, inputs, finalize_only=True, device="cpu")
    payload = json.loads((output / "generation_provenance.json").read_text(encoding="utf-8"))
    assert payload["status"] == "failed"


def test_finalize_only_rejects_wrong_size_png(tmp_path: Path) -> None:
    inputs = _generation_inputs(tmp_path)
    output = tmp_path / "outputs"
    _write_png(output / inputs["selection"]["expected_filenames"][0], size=(256, 512))
    _write_png(output / inputs["selection"]["expected_filenames"][1])
    with pytest.raises(ValueError, match="size_mismatch"):
        validate_or_prepare_output(output, inputs, finalize_only=True, device="cpu")
    payload = json.loads((output / "generation_provenance.json").read_text(encoding="utf-8"))
    assert payload["status"] == "failed"


def test_partial_resume_requires_matching_provenance(tmp_path: Path) -> None:
    inputs = _generation_inputs(tmp_path)
    output = tmp_path / "outputs"
    _write_png(output / inputs["selection"]["expected_filenames"][0])
    with pytest.raises(ValueError, match="no matching generation_provenance"):
        validate_or_prepare_output(output, inputs, finalize_only=False, device="cpu")

    _write_png(output / inputs["selection"]["expected_filenames"][1])
    validate_or_prepare_output(output, inputs, finalize_only=True, device="cpu")
    changed = _generation_inputs(tmp_path / "changed", generate_steps=8)
    with pytest.raises(ValueError, match="does not match current inputs"):
        validate_or_prepare_output(output, changed, finalize_only=False, device="cpu")


@pytest.mark.parametrize("finalize_only", [False, True])
@pytest.mark.parametrize(
    ("generation_git_commit", "generation_script_sha256", "expected_mode", "expected_source"),
    [
        (None, None, "live", "current_tool"),
        ("95e015e4568b0ec4cf6862204e8506f34c1d7366", None, "retrospective", "operator_supplied"),
        (None, "abc123", "retrospective", "operator_supplied"),
        ("95e015e4568b0ec4cf6862204e8506f34c1d7366", "abc123", "retrospective", "operator_supplied"),
    ],
)
def test_generation_code_provenance_canary(
    tmp_path: Path,
    generation_git_commit: str | None,
    generation_script_sha256: str | None,
    expected_mode: str,
    expected_source: str,
    finalize_only: bool,
) -> None:
    inputs = _generation_inputs(
        tmp_path,
        generation_git_commit=generation_git_commit,
        generation_script_sha256=generation_script_sha256,
    )
    output = tmp_path / "outputs"
    if finalize_only:
        for name in inputs["selection"]["expected_filenames"]:
            _write_png(output / name)
    payload = validate_or_prepare_output(output, inputs, finalize_only=finalize_only, device="cpu")
    generation = payload["code"]["generation"]
    assert generation["capture_mode"] == expected_mode
    assert generation["source"] == expected_source
    if generation_git_commit:
        assert generation["git_commit"] == generation_git_commit
    if generation_script_sha256:
        assert generation["script_sha256"] == generation_script_sha256
    assert payload["code"]["finalize_tool"]["source"] == "current_tool"
    signed = {key: value for key, value in inputs.items() if key != "input_signature_sha256"}
    signed["code"] = {key: value for key, value in signed["code"].items() if key != "finalize_tool"}
    assert canonical_sha256(signed) == inputs["input_signature_sha256"]


def test_finalize_only_accepts_legacy_signature_with_finalize_tool(tmp_path: Path) -> None:
    inputs = _generation_inputs(tmp_path)
    output = tmp_path / "outputs"
    for name in inputs["selection"]["expected_filenames"]:
        _write_png(output / name)

    legacy_body = dict(inputs)
    legacy_body["code"] = dict(inputs["code"])
    legacy_body["code"]["finalize_tool"] = {
        "path": "/old/tools/generate_role_flow_eval.py",
        "sha256": "old-tool-sha",
        "git_commit": "old-commit",
        "source": "current_tool",
    }
    legacy_body["input_signature_sha256"] = canonical_sha256(
        {key: legacy_body[key] for key in ("schema_version", "inputs", "selection", "generation", "code")}
    )
    (output / "generation_provenance.json").write_text(
        json.dumps({**legacy_body, "status": "complete", "images": [], "failures": []}),
        encoding="utf-8",
    )

    payload = validate_or_prepare_output(output, inputs, finalize_only=True, device="cpu")
    assert payload["status"] == "complete"
    assert payload["input_signature_sha256"] == inputs["input_signature_sha256"]
