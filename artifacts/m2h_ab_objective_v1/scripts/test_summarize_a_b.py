import unittest
from summarize_a_b import paired_effect, weights_for

class PairedTests(unittest.TestCase):
    def rows(self):
        return [{'mid':str(i),'jid':str(j),'seed':0,'m_source_group':str(i),'v':i+j}
                for i in range(4) for j in range(2)]

    def test_constant_difference_direction_and_ci(self):
        b=self.rows();a=[{**r,'v':r['v']+2} for r in b]
        result=paired_effect(a,b,'v')
        self.assertEqual(result['delta_mean'],2)
        self.assertEqual(result['ci95'],[2,2])
        self.assertEqual(result['bootstrap']['source_groups'],4)
        self.assertEqual(result['bootstrap']['reference_files'],2)

    def test_missing_not_silently_counted(self):
        b=self.rows();a=[{**r,'v':None} for r in b]
        result=paired_effect(a,b,'v')
        self.assertEqual(result['expected'],8);self.assertEqual(result['valid'],0)
        with self.assertRaises(ValueError): paired_effect(a[:-1],b,'v')

    def test_deterministic_cluster_weights(self):
        a,info=weights_for(self.rows(),100)
        b,_=weights_for(self.rows(),100)
        self.assertTrue((a==b).all())
        self.assertEqual(info['draws'],100)

if __name__=='__main__': unittest.main()
