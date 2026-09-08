import unittest
import numpy as np
from summarize_a2 import ARMS, REGIONS, validate, weights, estimate, effects, region_effects, widen_region_rows


def fixture():
    return [dict(mid=m,jid=j,seed=0,tau=0,k=1,branch=a,value=float(i),m_source_group=m)
            for i,a in enumerate(ARMS) for m,j in (('1','10'),('1','11'),('2','10'),('2','11'))]


class SummaryTests(unittest.TestCase):
    def test_validate(self):
        rr=fixture(); validate(rr,{(r['mid'],r['jid'],0) for r in rr})

    def test_missing_rejected(self):
        rr=fixture()
        with self.assertRaises(ValueError): validate(rr[:-1],{(r['mid'],r['jid'],0) for r in rr})

    def test_duplicate_rejected(self):
        rr=fixture()
        with self.assertRaises(ValueError): validate(rr+[rr[0]],{(r['mid'],r['jid'],0) for r in rr})

    def test_constant_effect(self):
        result=effects(fixture(),['value'])
        for name,delta in [('a2_minus_b2',1),('a4_minus_a2',1),('a4_minus_b2',2)]:
            self.assertEqual(result['contrasts'][name]['value']['mean'],delta)
            self.assertEqual(result['contrasts'][name]['value']['ci95'],[delta,delta])

    def test_nan_rejected(self):
        with self.assertRaises(ValueError): estimate([np.nan],np.ones((10,1)))

    def test_cross_reference_weights_reproducible(self):
        rr=[dict(mid='1',jid_left='10',jid_right='11'),dict(mid='2',jid_left='11',jid_right='12')]
        np.testing.assert_equal(weights(rr,50,cross=True),weights(rr,50,cross=True))
        self.assertEqual(estimate([2,2],weights(rr,50,cross=True))['ci95'],[2,2])

    def test_output_eligibility_rejected(self):
        rr=fixture()
        for r in rr: r.update(garment_eligible=True,garment_support=100)
        rr[0]['garment_eligible']=False
        with self.assertRaises(ValueError): region_effects(rr)

    def test_long_regions(self):
        rr=[dict(mid='1',jid='2',branch=ARMS[0],tau=0,k=1,seed=0,path='x',region=region,
                 metric_status='ok',eligible=True,support=10,region_dino=.9,region_hf_lpips=.1,
                 region_pixel_mae=.2,region_gradient_mae=.3) for region in REGIONS]
        wide=widen_region_rows(rr)
        self.assertEqual(wide[0]['interior_source_dino'],.9)
        self.assertEqual(wide[0]['boundary_support'],10)
        with self.assertRaises(ValueError): widen_region_rows(rr[:-1])
        with self.assertRaises(ValueError): widen_region_rows(rr+[rr[0]])

    def test_failed_region_cannot_be_hidden(self):
        r=dict(mid='1',jid='2',branch=ARMS[0],tau=0,k=1,seed=0,path='x',region='garment',metric_status='output_failed')
        with self.assertRaises(RuntimeError): widen_region_rows([r])


if __name__=='__main__': unittest.main()
