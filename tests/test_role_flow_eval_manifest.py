from __future__ import annotations

import csv
import json
from pathlib import Path

from tools.build_role_flow_eval_manifest import build_manifest
from tools.generate_role_flow_eval import load_manifest, select_rows


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
