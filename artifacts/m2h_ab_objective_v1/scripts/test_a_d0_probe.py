import numpy as np
import pytest
from a_d0_probe import make_state,other_reference
from build_diagnostic_cache import validate_pairs


def test_state_endpoints_and_shared_noise():
    a=np.ones(5); b=np.ones(5)*3; noise=np.ones(5)*7
    np.testing.assert_array_equal(make_state(a,noise,0),a)
    np.testing.assert_array_equal(make_state(a,noise,1),noise)
    np.testing.assert_allclose(make_state(a,noise,.4)-make_state(b,noise,.4),.6*(a-b))
    with pytest.raises(ValueError): make_state(a,noise,1.1)


def test_swap_preserves_endpoint_pool():
    rows=[{'mid':'1','jid':'2','seed':0},{'mid':'1','jid':'3','seed':0}]
    assert other_reference(rows,'1','2')=='3'
    assert other_reference(rows,'1','3')=='2'
    assert sorted(other_reference(rows,'1',r['jid']) for r in rows)==['2','3']
    assert validate_pairs(rows,{'1','2','3'})==(['1'],['2','3'])


def test_manifest_rejects_wrong_pairing():
    with pytest.raises(ValueError): validate_pairs([{'mid':'1','jid':'1','seed':0}],{'1'})
    with pytest.raises(ValueError): validate_pairs([{'mid':'1','jid':'2','seed':0}],{'1','2'})
