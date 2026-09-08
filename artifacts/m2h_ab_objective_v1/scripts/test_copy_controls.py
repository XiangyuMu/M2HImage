import numpy as np
from copy_controls import feather_copy, mask_regions, metrics


def test_copy_is_exact_only_in_reliable_interior():
    a = np.full((80,80,3),200,np.uint8)
    b = np.full_like(a,30)
    m = np.zeros((80,80),np.uint8)
    m[15:65,15:65]=1
    y = feather_copy(a,b,m,4,8)
    assert np.array_equal(y[~m.astype(bool)],b[~m.astype(bool)])
    assert np.array_equal(y[30:50,30:50],a[30:50,30:50])


def test_identity_metric_zero_and_empty_roi_explicit():
    a = np.zeros((80,80,3),np.uint8)
    m = np.zeros((80,80),np.uint8)
    r = metrics(a,a,a,m)
    assert r['interior_mse'] is None
    assert r['outside_allowed_edit_fraction']==0


def test_regions_partition():
    m=np.zeros((80,80),np.uint8)
    m[10:70,10:70]=1
    interior,band,exterior=mask_regions(m)
    assert np.all(interior.astype(int)+band+exterior==1)
