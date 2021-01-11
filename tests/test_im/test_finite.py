import numpy as np
import imcode.sparse as sparse
import imcode.shallow as shallow
import imcode.mps as mps
import imcode.dense as dense
import pytest
from ..utils import seed_rng
from .utils import check_dense_im

@pytest.fixture(scope="module")
def dense_L1_short_time():
    seed_rng("finite_L1")
    t,L=3,1
    dts=[]
    Js=np.random.normal(size=(L,))
    gs=np.random.normal(size=(L,))
    hs=np.random.normal(size=(L,))
    for i in range(L):
        dts.append(dense.ising_T(t,Js[i],gs[i],hs[i]))
    im=dense.im_finite(dts)
    return (im,(t,Js,gs,hs))
@pytest.fixture(scope="module")
def dense_L2_short_time():
    seed_rng("finite_L2")
    t,L=3,2
    dts=[]
    Js=np.random.normal(size=(L,))
    gs=np.random.normal(size=(L,))
    hs=np.random.normal(size=(L,))
    for i in range(L):
        dts.append(dense.ising_T(t,Js[i],gs[i],hs[i]))
    im=dense.im_finite(dts)
    return (im,(t,Js,gs,hs))
@pytest.fixture(scope="module")
def dense_L3_short_time():
    seed_rng("finite_L3")
    t,L=3,3
    dts=[]
    Js=np.random.normal(size=(L,))
    gs=np.random.normal(size=(L,))
    hs=np.random.normal(size=(L,))
    for i in range(L):
        dts.append(dense.ising_T(t,Js[i],gs[i],hs[i]))
    im=dense.im_finite(dts)
    return (im,(t,Js,gs,hs))

@pytest.fixture(scope="module")
def dense_L4_short_time():
    seed_rng("finite_L4")
    t,L=3,4
    dts=[]
    Js=np.random.normal(size=(L,))
    gs=np.random.normal(size=(L,))
    hs=np.random.normal(size=(L,))
    for i in range(L):
        dts.append(dense.ising_T(t,Js[i],gs[i],hs[i]))
    im=dense.im_finite(dts)
    return (im,(t,Js,gs,hs))
def test_dense_L4_short_time(dense_L4_short_time):
    check_dense_im(dense_L4_short_time[0])
def test_dense_L3_short_time(dense_L3_short_time):
    check_dense_im(dense_L3_short_time[0])
def test_dense_L2_short_time(dense_L2_short_time):
    check_dense_im(dense_L2_short_time[0])
def test_dense_L1_short_time(dense_L1_short_time):
    check_dense_im(dense_L1_short_time[0])

def test_sparse_L1_short_time(dense_L1_short_time):
    t=dense_L1_short_time[1][0]
    sts=[]
    for J,g,h in zip(*dense_L1_short_time[1][1:]):
        sts.append(sparse.ising_T(t,J,g,h))
    assert sparse.im_finite(sts)==pytest.approx(dense_L1_short_time[0])

def test_sparse_L2_short_time(dense_L2_short_time):
    t=dense_L2_short_time[1][0]
    sts=[]
    for J,g,h in zip(*dense_L2_short_time[1][1:]):
        sts.append(sparse.ising_T(t,J,g,h))
    assert sparse.im_finite(sts)==pytest.approx(dense_L2_short_time[0])
def test_sparse_L3_short_time(dense_L3_short_time):
    t=dense_L3_short_time[1][0]
    sts=[]
    for J,g,h in zip(*dense_L3_short_time[1][1:]):
        sts.append(sparse.ising_T(t,J,g,h))
    assert sparse.im_finite(sts)==pytest.approx(dense_L3_short_time[0])
def test_sparse_L4_short_time(dense_L4_short_time):
    t=dense_L4_short_time[1][0]
    sts=[]
    for J,g,h in zip(*dense_L4_short_time[1][1:]):
        sts.append(sparse.ising_T(t,J,g,h))
    assert sparse.im_finite(sts)==pytest.approx(dense_L4_short_time[0])

def test_mps_L1_short_time(dense_L1_short_time):
    t=dense_L1_short_time[1][0]
    sts=[]
    for J,g,h in zip(*dense_L1_short_time[1][1:]):
        sts.append(mps.ising_T(t,J,g,h))
    mif=mps.im_finite(sts)
    print(mps.mps_to_dense(mif)/mps.mps_to_dense(mif)[0])
    print(dense_L1_short_time[0])
    ps=dense_L1_short_time[1][1:]
    assert mps.mpo_to_dense(sts[0])==pytest.approx(dense.ising_T(t,ps[0][0],ps[1][0],ps[2][0]))

    assert mps.mps_to_dense(mps.im_finite(sts))==pytest.approx(dense_L1_short_time[0])
def test_mps_L2_short_time(dense_L2_short_time):
    t=dense_L2_short_time[1][0]
    sts=[]
    for J,g,h in zip(*dense_L2_short_time[1][1:]):
        sts.append(mps.ising_T(t,J,g,h))
    assert mps.mps_to_dense(mps.im_finite(sts))==pytest.approx(dense_L2_short_time[0])
def test_mps_L3_short_time(dense_L3_short_time):
    t=dense_L3_short_time[1][0]
    sts=[]
    for J,g,h in zip(*dense_L3_short_time[1][1:]):
        sts.append(mps.ising_T(t,J,g,h))
    assert mps.mps_to_dense(mps.im_finite(sts))==pytest.approx(dense_L3_short_time[0])
def test_mps_L4_short_time(dense_L4_short_time):
    t=dense_L4_short_time[1][0]
    sts=[]
    for J,g,h in zip(*dense_L4_short_time[1][1:]):
        sts.append(mps.ising_T(t,J,g,h))
    assert mps.mps_to_dense(mps.im_finite(sts))==pytest.approx(dense_L4_short_time[0])
def im_from_shallow(t,Js,gs,hs):
    def index_to_state(t,ind):
        return bin(ind+2**(2*t))[3:].replace("1","u").replace("0","d")
    ret=np.zeros((2**(2*t)),dtype=complex)
    for i in range(2**(2*t)):
        ret[i]=shallow.im_element(Js,gs,hs,index_to_state(t,i))
    return ret
def test_shallow_L1_short_time(dense_L1_short_time):
    assert im_from_shallow(*dense_L1_short_time[1])==pytest.approx(dense_L1_short_time[0])
def test_shallow_L2_short_time(dense_L2_short_time):
    assert im_from_shallow(*dense_L2_short_time[1])==pytest.approx(dense_L2_short_time[0])
def test_shallow_L3_short_time(dense_L3_short_time):
    assert im_from_shallow(*dense_L3_short_time[1])==pytest.approx(dense_L3_short_time[0])
def test_shallow_L4_short_time(dense_L4_short_time):
    assert im_from_shallow(*dense_L4_short_time[1])==pytest.approx(dense_L4_short_time[0])

def test_mps_shallow_L1_long_time():
    pass
def test_mps_shallow_L2_long_time():
    pass
def test_mps_shallow_L3_long_time():
    pass
def test_mps_shallow_L4_long_time():
    pass
