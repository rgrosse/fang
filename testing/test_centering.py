import gnumpy as gnp
import numpy as np

import centering

NVIS = 8
NHID = 6
NDATA = 10

def unif(size):
    return gnp.garray(np.random.uniform(0., 1., size=size))

def test_product_err():
    v = gnp.garray(np.random.binomial(1, 0.5, size=(NDATA, NVIS)))
    h = gnp.garray(np.random.binomial(1, 0.5, size=(NDATA, NHID)))
    expect = centering.RCTExpectations(unif(NVIS), unif(NHID), unif((NVIS, NHID)))
    err = centering.RCTErr.from_activations(v, h, expect)

    total_err = np.zeros((NVIS, NHID))
    for i in range(NVIS):
        for j in range(NHID):
            for d in range(NDATA):
                pred = expect.prod[i, j]
                actual = (v[d, i] - expect.vis[i]) * (h[d, j] - expect.hid[j])
                total_err[i, j] += (pred - actual) ** 2
    avg_err = total_err / NDATA

    assert np.allclose(err.prod.as_numpy_array(), avg_err)
    
