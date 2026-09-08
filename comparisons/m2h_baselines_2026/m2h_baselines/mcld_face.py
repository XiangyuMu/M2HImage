from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


FACE_CONTEXT_PADDING = 0.28125


def _with_detection_context(image: np.ndarray) -> np.ndarray:
    """Add neutral context around the dataset's tightly aligned face crop."""
    height, width = image.shape[:2]
    pad_y = int(round(height * FACE_CONTEXT_PADDING))
    pad_x = int(round(width * FACE_CONTEXT_PADDING))
    return cv2.copyMakeBorder(
        image,
        pad_y,
        pad_y,
        pad_x,
        pad_x,
        cv2.BORDER_CONSTANT,
        value=(127, 127, 127),
    )


def build_embeddings(
    layout_root: Path,
    phase: str,
    model_root: Path,
    limit: int | None = None,
    device: str = "cpu",
) -> dict[str, object]:
    from insightface.app import FaceAnalysis

    input_dir = layout_root / f"{phase}_face_inputs"
    output_dir = layout_root / f"{phase}_face"
    output_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(input_dir.glob("*.png"))
    if limit is not None:
        images = images[:limit]
    if device == "cuda":
        providers: list[object] = [
            ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),
            "CPUExecutionProvider",
        ]
    elif device == "cpu":
        providers = ["CPUExecutionProvider"]
    else:
        raise ValueError(f"unsupported face embedding device: {device}")
    app = FaceAnalysis(
        name="antelopev2",
        root=str(model_root),
        # MCLD consumes only the 512-D recognition embedding.  Loading and
        # running landmark/attribute heads changes no embedding values and is
        # prohibitively expensive for the 36k-sample full split.
        allowed_modules=["detection", "recognition"],
        providers=providers,
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    failures: list[dict[str, str]] = []
    fallback_counts = {"lower_threshold": 0, "identity_card": 0}
    written = 0
    for image_path in images:
        output_path = output_dir / f"{image_path.stem}.npy"
        if output_path.exists():
            continue
        image = cv2.imread(str(image_path))
        faces = app.get(_with_detection_context(image)) if image is not None else []
        if not faces and image is not None:
            app.det_model.det_thresh = 0.2
            faces = app.get(_with_detection_context(image))
            if faces:
                fallback_counts["lower_threshold"] += 1
        if not faces:
            # The MCLD layout's identity image is built from the same jid and
            # is therefore a valid identity-only fallback.  Counterfactual
            # target human(mid) is never read here.
            identity_card = layout_root / phase / image_path.name
            card = cv2.imread(str(identity_card))
            faces = app.get(card) if card is not None else []
            if faces:
                fallback_counts["identity_card"] += 1
        app.det_model.det_thresh = 0.5
        if not faces:
            failures.append({"image": str(image_path), "error": "no_face"})
            continue
        face = max(
            faces,
            key=lambda item: float(
                (item["bbox"][2] - item["bbox"][0])
                * (item["bbox"][3] - item["bbox"][1])
            ),
        )
        embedding = np.asarray(face["embedding"], dtype=np.float32)
        if embedding.shape != (512,) or not np.isfinite(embedding).all():
            failures.append({"image": str(image_path), "error": f"bad_embedding:{embedding.shape}"})
            continue
        np.save(output_path, embedding)
        written += 1
    report = {
        "phase": phase,
        "device": device,
        "requested": len(images),
        "written": written,
        "available": len(list(output_dir.glob("*.npy"))),
        "fallback_counts": fallback_counts,
        "failures": failures,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if failures:
        raise RuntimeError(f"MCLD face embedding failed for {len(failures)} samples")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build official Antelopev2 embeddings for MCLD M2H layouts.")
    parser.add_argument("--layout-root", required=True)
    parser.add_argument("--phase", choices=("train", "test"), required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    report = build_embeddings(
        Path(args.layout_root),
        args.phase,
        Path(args.model_root),
        args.limit,
        args.device,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
