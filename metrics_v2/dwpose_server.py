from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm


def load_dwpose_functions(repo_root: Path):
    """Load the two ONNX helpers from the common DWPose repository layouts."""
    candidates = (
        ("dwpose_utils.onnxdet", "dwpose_utils.onnxpose"),
        ("dwpose.onnxdet", "dwpose.onnxpose"),
        ("annotator.dwpose.onnxdet", "annotator.dwpose.onnxpose"),
    )
    errors: list[str] = []
    sys.path.insert(0, str(repo_root))
    try:
        for detector_module, pose_module in candidates:
            try:
                detector = importlib.import_module(detector_module)
                pose = importlib.import_module(pose_module)
                return detector.inference_detector, pose.inference_pose
            except (ImportError, AttributeError) as exc:
                errors.append(f"{detector_module}/{pose_module}: {exc}")
    finally:
        sys.path.pop(0)
    details = "\n  - ".join(errors)
    raise ImportError(
        f"unable to import DWPose ONNX helpers from {repo_root}; tried:\n  - {details}"
    )


class DWPoseBatchModel:
    def __init__(self, repo_root: Path, detector: Path, pose: Path, provider: str):
        inference_detector, inference_pose = load_dwpose_functions(repo_root)
        providers = [provider]
        if provider != "CPUExecutionProvider":
            providers.append("CPUExecutionProvider")
        self.detector = ort.InferenceSession(str(detector), providers=providers)
        self.pose = ort.InferenceSession(str(pose), providers=providers)
        self.inference_detector = inference_detector
        self.inference_pose = inference_pose

    def __call__(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        detections = self.inference_detector(self.detector, rgb)
        keypoints, scores = self.inference_pose(self.pose, detections, rgb)
        if keypoints.shape[0] == 0:
            raise RuntimeError("DWPose found no person")
        info = np.concatenate((keypoints, scores[..., None]), axis=-1)
        neck = np.mean(info[:, [5, 6]], axis=1)
        neck[:, 2] = np.logical_and(info[:, 5, 2] > 0.3, info[:, 6, 2] > 0.3).astype(np.float32)
        info = np.insert(info, 17, neck, axis=1)
        mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
        openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
        info[:, openpose_idx] = info[:, mmpose_idx]
        body_scores = info[:, :18, 2]
        person_scores = np.where(body_scores > 0.3, body_scores, 0.0).sum(axis=1)
        selected = int(np.argmax(person_scores))
        return info[selected, :18, :2].astype(np.float32), body_scores[selected].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch DWPose helper for metrics_v2.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--pose", required=True)
    parser.add_argument("--provider", default="CUDAExecutionProvider")
    args = parser.parse_args()

    requests = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    model = DWPoseBatchModel(Path(args.repo_root), Path(args.detector), Path(args.pose), args.provider)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for request in tqdm(requests, desc="DWPose generated images", file=sys.stderr):
            result = {"key": request["key"], "path": request["path"], "status": "ok", "error": ""}
            try:
                bgr = cv2.imread(request["path"], cv2.IMREAD_COLOR)
                if bgr is None:
                    raise RuntimeError("failed to read image")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                body, scores = model(rgb)
                body[:, 0] /= float(rgb.shape[1])
                body[:, 1] /= float(rgb.shape[0])
                result.update(
                    {
                        "body": body.tolist(),
                        "body_scores": scores.tolist(),
                        "width": int(rgb.shape[1]),
                        "height": int(rgb.shape[0]),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                result["status"] = "failed"
                result["error"] = str(exc)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()


if __name__ == "__main__":
    main()
