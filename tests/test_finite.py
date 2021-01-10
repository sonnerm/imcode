import numpy as np
import imcode.sparse as sparse
import imcode.shallow as shallow
import imcode.mps as mps
import imcode.dense as dense
import pytest
from .utils import seed_rng

@pytest.fixture(scope="module")
def dense_L1_short_time():
    seed_rng("finite_L1")
    t,L=3,1
    dts=[]
    for _ in range(L):
        J=np.random.normal()
        g=np.random.normal()
        h=np.random.normal()
        dts.append(dense.ising_T(t,J,g,h))
    im=dense.im_finite(dts)
    return (im,(t,J,g,h))
@pytest.fixture(scope="module")
def dense_L2_short_time():
    seed_rng("finite_L2")
    t,L=3,2
    dts=[]
    for _ in range(L):
        J=np.random.normal()
        g=np.random.normal()
        h=np.random.normal()
        dts.append(dense.ising_T(t,J,g,h))
    im=dense.im_finite(dts)
    return (im,(t,J,g,h))
@pytest.fixture(scope="module")
def dense_L3_short_time():
    seed_rng("finite_L3")
    t,L=3,3
    dts=[]
    for _ in range(L):
        J=np.random.normal()
        g=np.random.normal()
        h=np.random.normal()
        dts.append(dense.ising_T(t,J,g,h))
    im=dense.im_finite(dts)
    return (im,(t,J,g,h))

@pytest.fixture(scope="module")
def dense_L4_short_time():
    seed_rng("finite_L4")
    t,L=3,3
    dts=[]
    for _ in range(L):
        J=np.random.normal()
        g=np.random.normal()
        h=np.random.normal()
        dts.append(dense.ising_T(t,J,g,h))
    im=dense.im_finite(dts)
    return (im,(t,J,g,h))
def check_dense_im(im):
    pass
def test_dense_L4_short_time(dense_L4_short_time):
    check_dense_im(dense_L4_short_time[0])
def test_dense_L3_short_time(dense_L3_short_time):
    check_dense_im(dense_L3_short_time[0])
def test_dense_L2_short_time(dense_L2_short_time):
    check_dense_im(dense_L2_short_time[0])
def test_dense_L1_short_time(dense_L1_short_time):
    check_dense_im(dense_L1_short_time[0])

def test_sparse_L1_short_time(dense_L1_short_time):
    pass
