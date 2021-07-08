import imcode.mps as mps
import numpy as np
import pytest

def test_open_boundary_channel(seed_rng):
    t=4
    init=np.random.normal(size=(2,2))+np.random.normal(size=(2,2))*1.0j
    init=init+init.T.conj()
    im=mps.ising.open_boundary_im(t)
    for i in range(t):
        imc=mps.ising.im_channel_dense(im,i)
        assert imc==pytest.approx(np.eye(4))
