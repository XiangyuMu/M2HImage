from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from tools.build_role_flow_metric_protocol import (
    SampleRecord,
    build_fid_reference_manifest,
    build_tar_calibration,
    calibration_threshold,
    compute_impostor_scores,
    compute_impostor_scores_for_records,
    impostor_pair_count,
    impostor_pairs,
    load_person_manifest,
    load_split_records,
)


class FakeEmbedder:
    metadata = {
        "recognizer": "fake-adaface",
        "checkpoint_sha256": "0" * 64,
        "detector": "fake-detector",
        "preprocess": {"aligned_size": 112, "normalize": "l2"},
    }

    def __init__(self, embeddings: dict[str, np.ndarray]):
        self.embeddings = embeddings

    def embed(self, face_crop_path: Path) -> np.ndarray:
        return self.embeddings[face_crop_path.stem]


def write_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    dataset = tmp_path / "clean"
    asset = tmp_path / "asset"
    report = dataset / "dataset_cleaning_report"
    split_dir = dataset / "splits_person_disjoint_final"
    human_dir = asset / "human"
    face_dir = asset / "derived" / "face_crops" / "human"
    for path in (report, split_dir, human_dir, face_dir):
        path.mkdir(parents=True)
    rows = [
        ("t2", "val", "ct2", "src", "kt2"),
        ("t1", "test", "ct1", "src", "kt1"),
        ("v1", "val", "cv1", "src", "kv1"),
        ("v2", "val", "cv2", "src", "kv2"),
        ("v3", "val", "cv2", "src", "kv3"),
        ("q1", "quarantine", "cq1", "src", "kq1"),
    ]
    for sample_id, _, _, _, _ in rows:
        (human_dir / f"{sample_id}.jpg").write_bytes(f"human-{sample_id}".encode("utf-8"))
        (face_dir / f"{sample_id}.png").write_bytes(f"face-{sample_id}".encode("utf-8"))
    person = report / "person_disjoint_manifest.csv"
    with person.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "human_path", "clean_split", "person_cluster_id", "source_dataset", "source_key"],
        )
        writer.writeheader()
        for sample_id, split, cluster, source_dataset, source_key in rows:
            writer.writerow({
                "id": sample_id,
                "human_path": str(human_dir / f"{sample_id}.jpg"),
                "clean_split": split,
                "person_cluster_id": cluster,
                "source_dataset": source_dataset,
                "source_key": source_key,
            })
    (split_dir / "train.txt").write_text("", encoding="utf-8")
    (split_dir / "val.txt").write_text("v3\nv1\nv2\n", encoding="utf-8")
    (split_dir / "test.txt").write_text("t2\nt1\n", encoding="utf-8")
    (split_dir / "quarantine.txt").write_text("q1\n", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text("metrics:\n  heldout_id:\n    recognizer: fake\n", encoding="utf-8")
    return dataset, asset, person, config


def unit(index: int) -> np.ndarray:
    value = np.zeros((512,), dtype=np.float32)
    value[index] = 1.0
    return value


def test_fid_reference_manifest_freezes_sorted_final_test_humans(tmp_path: Path) -> None:
    dataset, asset, person, config = write_fixture(tmp_path)
    output = tmp_path / "fid.json"

    payload = build_fid_reference_manifest(
        dataset_root=dataset,
        asset_root=asset,
        person_manifest_path=person,
        split_dir=dataset / "splits_person_disjoint_final",
        config_path=config,
        output_path=output,
    )

    assert payload["count"] == 2
    assert [row["id"] for row in payload["rows"]] == ["t1", "t2"]
    assert all(row["split"] == "test" and row["human_sha256"] for row in payload["rows"])
    assert payload["rows"][1]["manifest_clean_split"] == "val"
    assert payload["split_authority"].startswith("splits_person_disjoint_final")
    assert payload["input_sha256"]["person_manifest"]
    assert payload["input_sha256"]["split_counts"]["test"] == 2
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["content_sha256"] == payload["content_sha256"]


def test_tar_calibration_uses_all_unordered_impostors_except_same_cluster(tmp_path: Path) -> None:
    dataset, asset, person, config = write_fixture(tmp_path)
    output = tmp_path / "tar.json"
    embeddings_npz = tmp_path / "embeddings.npz"
    embedder = FakeEmbedder({"v1": unit(0), "v2": unit(0), "v3": unit(1)})

    payload = build_tar_calibration(
        dataset_root=dataset,
        asset_root=asset,
        person_manifest_path=person,
        split_dir=dataset / "splits_person_disjoint_final",
        config_path=config,
        output_path=output,
        embedding_npz_path=embeddings_npz,
        embedder=embedder,
        far=1e-3,
    )

    assert payload["val_id_count"] == 3
    assert payload["impostor_pair_count"] == 2
    assert payload["excluded_same_cluster_pair_count"] == 1
    assert payload["threshold"] > 1.0
    assert payload["empirical_far"] == pytest.approx(0.0)
    assert payload["allowed_false_accept_count"] == 0
    assert payload["embedding_npz_sha256"]
    stored = np.load(embeddings_npz)
    assert stored["embeddings"].shape == (3, 512)


def test_tar_calibration_fails_closed_on_missing_embedding(tmp_path: Path) -> None:
    dataset, asset, person, config = write_fixture(tmp_path)
    embedder = FakeEmbedder({"v1": unit(0), "v2": unit(0)})

    with pytest.raises(RuntimeError, match="TAR calibration failed"):
        build_tar_calibration(
            dataset_root=dataset,
            asset_root=asset,
            person_manifest_path=person,
            split_dir=dataset / "splits_person_disjoint_final",
            config_path=config,
            output_path=tmp_path / "tar.json",
            embedding_npz_path=tmp_path / "embeddings.npz",
            embedder=embedder,
        )
    payload = json.loads((tmp_path / "tar.json").read_text(encoding="utf-8"))
    assert payload["status"] == "blocked"
    assert payload["failure_count"] == 1
    assert payload["failures"][0]["id"] == "v3"


def test_protocol_pure_helpers_reject_bad_inputs(tmp_path: Path) -> None:
    threshold, empirical_far = calibration_threshold(np.asarray([0.1, 0.2, 0.3], dtype=np.float32), 1e-3)
    assert threshold > 0.3
    assert empirical_far == pytest.approx(0.0)
    threshold, empirical_far = calibration_threshold(
        np.asarray([0.9, 0.8, 0.8, 0.1], dtype=np.float32),
        0.5,
    )
    assert threshold > 0.8
    assert empirical_far == pytest.approx(0.25)
    with pytest.raises(ValueError, match="at least one impostor"):
        calibration_threshold(np.asarray([], dtype=np.float32), 1e-3)
    with pytest.raises(ValueError, match="far must be"):
        calibration_threshold(np.asarray([0.1], dtype=np.float32), -0.1)
    with pytest.raises(ValueError, match="missing columns"):
        path = tmp_path / "bad.csv"
        path.write_text("id,human_path\nx,/tmp/x.jpg\n", encoding="utf-8")
        load_person_manifest(path)


def test_impostor_scores_follow_record_order_and_cluster_exclusion(tmp_path: Path) -> None:
    dataset, _, person, _ = write_fixture(tmp_path)
    manifest = load_person_manifest(person)
    records = load_split_records(
        split_name="val",
        split_dir=dataset / "splits_person_disjoint_final",
        person_manifest=manifest,
    )
    pairs = impostor_pairs(records)
    scores = compute_impostor_scores(np.stack([unit(0), unit(0), unit(1)]), pairs)
    matrix_scores, excluded_count = compute_impostor_scores_for_records(np.stack([unit(0), unit(0), unit(1)]), records)
    assert pairs == [(0, 1), (0, 2)]
    assert scores.tolist() == [1.0, 0.0]
    assert matrix_scores.tolist() == [1.0, 0.0]
    assert excluded_count == 1


def test_singleton_1955_val_population_pair_count() -> None:
    records = [
        SampleRecord(sample_id=f"v{index:04d}", row={"person_cluster_id": f"c{index:04d}"})
        for index in range(1955)
    ]

    assert impostor_pair_count(records) == 1_910_035
