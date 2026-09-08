import json
import tempfile
from pathlib import Path
import unittest
import numpy as np
from revalidation_common import load_pairs, validate_payload, SOURCE_KEYS, IDENTITY_KEYS, image_path


class Contracts(unittest.TestCase):
    def test_duplicate_pairs_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'pairs.json'
            r = {'mid': '001', 'jid': '002', 'seed': 0}
            p.write_text(json.dumps({'pairs': [r, r]}))
            with self.assertRaises(ValueError): load_pairs(p)

    def test_role_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'pairs.json'
            p.write_text(json.dumps({'pairs': [{'mid': '001', 'jid': '001', 'seed': 0}]}))
            with self.assertRaises(ValueError): load_pairs(p)

    def test_valid_pairs(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'pairs.json'
            p.write_text(json.dumps({'pairs': [{'mid': '001', 'jid': '002', 'seed': 0}]}))
            self.assertEqual(len(load_pairs(p)), 1)

    def test_target_payload_rejected(self):
        payload = {k: np.zeros((7,)) for k in SOURCE_KEYS}
        validate_payload(payload, SOURCE_KEYS)
        payload['target_latents'] = np.zeros((1,))
        with self.assertRaises(ValueError): validate_payload(payload, SOURCE_KEYS)

    def test_nonfinite_rejected(self):
        payload = {k: np.zeros((7,)) for k in SOURCE_KEYS}
        payload['head_pose'][0] = np.nan
        with self.assertRaises(ValueError): validate_payload(payload, SOURCE_KEYS)

    def test_reference_shape(self):
        p = {'appearance': np.zeros(1024), 'pulid_id_embed': np.zeros((32, 2048))}
        validate_payload(p, IDENTITY_KEYS)
        p['pulid_id_embed'] = np.zeros(512)
        with self.assertRaises(ValueError): validate_payload(p, IDENTITY_KEYS)

    def test_role_path_denied(self):
        with self.assertRaises(ValueError): image_path('/tmp', 'derived/head_pose_6drepnet/human', '001')


if __name__ == '__main__': unittest.main()
