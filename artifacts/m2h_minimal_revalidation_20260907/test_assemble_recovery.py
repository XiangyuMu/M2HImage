import unittest
from assemble_recovery_v2 import combine,key


class BridgeMerge(unittest.TestCase):
    def setUp(self):
        self.old=[{'mid':str(i),'jid':str(i+100),'seed':0,'branch':m+'_legacyM_oldI'} for m in ('a4','b2') for i in range(8)]
        self.fresh=[{**r,'branch':r['branch'].replace('_oldI','_freshI')} for r in self.old]
        self.expected={key(r) for r in self.old}

    def test_full_merge(self):
        self.assertEqual(len(combine(self.old,self.fresh,self.expected)),32)

    def test_missing_counterpart_rejected(self):
        with self.assertRaises(ValueError):combine(self.old,self.fresh[:-1],self.expected)

    def test_duplicate_counterpart_rejected(self):
        with self.assertRaises(ValueError):combine(self.old,self.fresh+[self.fresh[0]],self.expected)

    def test_wrong_seed_rejected(self):
        self.fresh[0]['seed']=1
        with self.assertRaises(ValueError):combine(self.old,self.fresh,self.expected)

    def test_extra_main_rows_not_selected(self):
        extra={**self.fresh[0],'mid':'unrelated'}
        self.assertEqual(len(combine(self.old,self.fresh+[extra],self.expected)),32)


if __name__=='__main__':unittest.main()
