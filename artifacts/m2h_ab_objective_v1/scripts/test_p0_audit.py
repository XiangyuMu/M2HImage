import unittest
from p0_audit import source_group, summarize, select_old_val


class AuditTests(unittest.TestCase):
    def rows(self):
        return [dict(id='1', source_dataset='x', source_key='item_1', split='train'),
                dict(id='2', source_dataset='x', source_key='item_2', split='val'),
                dict(id='3', source_dataset='y', source_key='item_3', split='test')]

    def test_parent_risk_not_duplicate_claim(self):
        report, overlap = summarize(self.rows(), {'train': {'1'}, 'val': {'2'}, 'test': {'3'}})
        self.assertEqual(report['cross_split_parent_candidate_groups'], 1)
        self.assertEqual(len(overlap['x:item']), 2)
        self.assertFalse(report['formal_split_ready'])
        self.assertFalse(report['split_manifest_mismatch_ids'])

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            summarize([self.rows()[0]] * 2, {'train': {'1'}})

    def test_mismatch_and_extra_ids(self):
        report, _ = summarize(self.rows(), {'train': {'1', '2', '9'}, 'test': {'3'}})
        self.assertEqual(report['split_manifest_mismatch_ids'], ['2'])
        self.assertEqual(report['split_ids_missing_manifest'], ['9'])

    def test_probe_is_deterministic_val_only(self):
        self.assertEqual(select_old_val(self.rows(), 64), ['2'])
        self.assertEqual(select_old_val(list(reversed(self.rows())), 64), ['2'])

    def test_non_numeric_suffix_not_truncated(self):
        self.assertEqual(source_group({'source_dataset': 'x', 'source_key': 'a_b'}), 'x:a_b')


if __name__ == '__main__':
    unittest.main()
