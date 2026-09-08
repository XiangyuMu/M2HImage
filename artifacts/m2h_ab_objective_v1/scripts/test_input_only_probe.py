import unittest
import tempfile
from pathlib import Path
import numpy as np
from input_only_probe import validate_ids, identity_payload, source_path, check_dataset_read


class InputContractTests(unittest.TestCase):
    def test_holdout_and_path_rejected(self):
        for ids in (['1', '3'], ['1', '../2'], ['1', '1']):
            with self.assertRaises(ValueError):
                validate_ids(ids, {'1', '2'})
        validate_ids(['1', '2'], {'1', '2'})

    def test_identity_selection_excludes_target_and_garment(self):
        data = {key: np.ones(2) for key in ('pulid_id_embed', 'appearance', 'target_latents', 'head_pose', 'garment_grid')}
        self.assertEqual(set(identity_payload(data)), {'pulid_id_embed', 'appearance'})

    def test_target_role_denied(self):
        with self.assertRaises(ValueError):
            source_path('/unused', 'images/human', '1')

    def test_symlink_escape_denied(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'images/mannequin'
            source.mkdir(parents=True)
            human = root / 'target.png'
            human.touch()
            (source / '1.png').symlink_to(human)
            with self.assertRaises(ValueError):
                source_path(root, 'images/mannequin', '1')

    def test_target_and_unlisted_cache_denied(self):
        root = Path('/dataset')
        allowed = {root / 'images/mannequin/1.png'}
        check_dataset_read(root / 'images/mannequin/1.png', root, allowed)
        for path in ('images/human/1.jpg', 'phase1/old_cache/1.npz'):
            with self.assertRaises(PermissionError):
                check_dataset_read(root / path, root, allowed)


if __name__ == '__main__':
    unittest.main()
