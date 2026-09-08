import copy
import unittest

from freeze_data_protocol import source_group, validate_pairs, duplicate_risk, select_dev_pairs

try:
    import numpy as np
except ImportError:
    np = None


class SchemaTests(unittest.TestCase):
    def test_source_semantics(self):
        def key(value):
            return source_group({'source_dataset': 'z', 'source_key': value})
        self.assertEqual(key('multiimages_x2__00001_1'), key('multiimages_x2__00001_2'))
        self.assertNotEqual(key('singleimage_x2__00001'), key('singleimage_x2__00002'))
        self.assertNotEqual(key('train__00001_00'), key('test__00001_00'))
        self.assertEqual(key('unknown_123'), 'z:unknown_123')

    def test_pairs_contract_and_donor(self):
        payload = {'pairs': [{'mid': '00001', 'jid': '00002', 'seed': 0, 'donor_jid': '00003'},
                             {'mid': '00001', 'jid': '00003', 'seed': 0, 'donor_jid': '00002'}]}
        validate_pairs(payload, 1)
        for field, value in [('jid', '00001'), ('jid', '00002'), ('donor_jid', '00004'), ('seed', 1)]:
            bad = copy.deepcopy(payload)
            bad['pairs'][1][field] = value
            with self.assertRaises(ValueError):
                validate_pairs(bad, 1)

    def test_hashes_cross_roles_and_nonzero_hamming(self):
        def record(h, d):
            return {'sha256': h, 'dhash': f'{d:016x}'}
        audit = {'a': {'human': record('a', 0), 'mannequin': record('b', 0)},
                 'b': {'human': record('c', 31), 'mannequin': record('d', 31)}}
        self.assertFalse(duplicate_risk('a', 'b', audit))
        audit['b']['human']['dhash'] = '000000000000000f'
        self.assertTrue(duplicate_risk('a', 'b', audit))
        audit['b']['human']['dhash'] = 'ffffffffffffffff'
        audit['b']['mannequin']['sha256'] = 'a'
        self.assertTrue(duplicate_risk('a', 'b', audit))


@unittest.skipIf(np is None, 'NumPy unavailable locally; run numeric tests in existing remote CPU environment')
class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.ids = [f'{i:05d}' for i in range(1, 9)]
        self.rows = [{'id': s, 'source_dataset': 'z', 'source_key': 'singleimage_x2__'+s, 'split': 'val'} for s in self.ids]
        self.vectors = np.eye(8)
        hashes = [0x0000000000000000, 0xffffffffffffffff, 0xaaaaaaaaaaaaaaaa, 0x5555555555555555,
                  0xcccccccccccccccc, 0x3333333333333333, 0xf0f0f0f0f0f0f0f0, 0x0f0f0f0f0f0f0f0f]
        self.audit = {s: {role: {'sha256': s+role, 'dhash': f'{hashes[i]:016x}'}
                          for role in ('human', 'mannequin')} for i, s in enumerate(self.ids)}

    def select(self):
        return select_dev_pairs(self.rows, self.ids[:2], self.ids, self.vectors, self.audit, expected_m=2)

    def test_deterministic_disjoint_two_refs(self):
        a = self.select()
        self.assertEqual(a, self.select())
        validate_pairs(a, 2)
        self.assertEqual(a['mids'], self.ids[:2])
        self.assertLess(a['pair_max_raw_glint_cosine'], .3)
        self.assertFalse(a['formal_split_ready'])
        order = list(reversed(range(8)))
        b = select_dev_pairs(list(reversed(self.rows)), self.ids[:2], list(reversed(self.ids)),
                             self.vectors[order], self.audit, expected_m=2)
        self.assertEqual(a, b)

    def test_source_and_hash_risks_excluded(self):
        self.rows[2]['source_key'] = self.rows[0]['source_key']
        self.audit[self.ids[3]]['human'] = self.audit[self.ids[1]]['human'].copy()
        result = self.select()
        self.assertFalse(set(self.ids[2:4]) & set(result['reference_ids']))

    def test_full_vectors_reject_high_similarity(self):
        self.vectors[2:] = self.vectors[0]
        with self.assertRaisesRegex(ValueError, 'No admissible'):
            self.select()

    def test_invalid_vectors_fail_closed(self):
        for value in (float('nan'), 0, 2):
            self.vectors = np.eye(8)
            self.vectors[0, 0] = value
            with self.assertRaises(ValueError):
                self.select()


if __name__ == '__main__':
    unittest.main()
