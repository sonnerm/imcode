import imcode.mps as mps
import numpy as np
import pytest

def test_open_boundary_channel(seed_rng):
    t=4
    im=mps.ising.open_boundary_im(t)
    for i in range(t):
        imc=mps.ising.im_channel_dense(im,i)
        assert imc==pytest.approx(np.eye(4))

def test_pd_channel():
    t=4
    im=mps.ising.perfect_dephaser_im(t)
    for i in range(t):
        imc=mps.ising.im_channel_dense(im,i)
        assert imc==pytest.approx(np.diag([1.0,0.0,0.0,1.0]))

def test_dephaser_channel(seed_rng):
    t=4
    gamma=np.random.uniform(0,1)
    im=mps.ising.dephaser_im(t,gamma)
    for i in range(t):
        imc=mps.ising.im_channel_dense(im,i)
        assert imc==pytest.approx(np.diag([1.0,1-gamma,1-gamma,1.0]))
