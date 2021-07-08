import imcode.mps as mps
import imcode.dense as dense
import scipy.linalg as la
import numpy as np
import pytest

def test_open_boundary_channel(seed_rng):
    t=4
    init=np.random.normal(size=(2,2))+np.random.normal(size=(2,2))*1.0j
    init=init+init.T.conj()
    im=mps.ising.open_boundary_im(t)
    imc=mps.ising.im_channel(im)
