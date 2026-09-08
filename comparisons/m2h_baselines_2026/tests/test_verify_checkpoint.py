import json
import tempfile
import unittest
from pathlib import Path

from m2h_baselines.verify_checkpoint import verify_checkpoint


PROTOCOL = "a" * 64


class VerifyCheckpointTest(unittest.TestCase):
    def _metadata(self, root: Path, **extra: object) -> None:
        payload: dict[str, object] = {
            "task": "mannequin_to_human",
            "global_step": 100,
            "protocol_sha256": PROTOCOL,
            "conditioning_contract": {"source": "mannequin(mid)"},
        }
        payload.update(extra)
        (root / "m2h_checkpoint.json").write_text(json.dumps(payload))

    def test_idm_requires_task_provenance_and_nonempty_unet(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "unet").mkdir()
            (root / "unet/diffusion_pytorch_model.safetensors").write_bytes(b"model")
            self._metadata(root)
            report = verify_checkpoint(root, method="idm_vton", protocol_sha256=PROTOCOL)
            self.assertEqual(report["global_step"], 100)

            self._metadata(root, task="virtual_try_on")
            with self.assertRaisesRegex(ValueError, "mannequin_to_human"):
                verify_checkpoint(root, method="idm_vton", protocol_sha256=PROTOCOL)

    def test_mcld_and_omini_require_all_declared_files(self) -> None:
        cases = {
            "mcld": [
                "reference_unet-100.pth",
                "denoising_unet-100.pth",
                "pose_guider-100.pth",
                "image_proj_model-100.pth",
            ],
            "ominicontrol": [
                "mannequin.safetensors",
                "identity.safetensors",
                "pose.safetensors",
            ],
        }
        for method, files in cases.items():
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for name in files:
                    (root / name).write_bytes(b"weights")
                field = "component_files" if method == "mcld" else "adapter_files"
                self._metadata(root, **{field: files})
                report = verify_checkpoint(root, method=method, protocol_sha256=PROTOCOL)
                self.assertEqual(set(report["files"]), set(files))

                (root / files[-1]).unlink()
                with self.assertRaises(FileNotFoundError):
                    verify_checkpoint(root, method=method, protocol_sha256=PROTOCOL)


if __name__ == "__main__":
    unittest.main()
