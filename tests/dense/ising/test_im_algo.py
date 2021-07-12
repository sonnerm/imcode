import numpy as np
import imcode.dense as dense
import pytest
def test_im_rectangle(seed_rng):
    J,g,h=np.random.normal(size=3)
    t=5
    T=dense.ising.ising_T(t,J,g,h)
    ims=[im for im in dense.ising.im_rectangle(T)]
    assert ims[-1]==pytest.approx(ims[-2]) #convergence achieved
    assert ims[-1]!=pytest.approx(ims[1]) #but not immediately
    assert ims[-1]==pytest.approx(dense.ising.im_diag(T)) #correct ev

def test_im_diamond(seed_rng):
    J,g,h=np.random.normal(size=3)
    t=5
    Ts=[dense.ising.ising_T(t,J,g,h) for t in range(1,6,2)]
    ims=[im for im in dense.ising.im_diamond(Ts)]
    for im,T in zip(ims,Ts):
        assert im ==pytest.approx(dense.ising.im_diag(T))
