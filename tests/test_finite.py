import numpy as np
import imcode.sparse as sparse
import imcode.shallow as shallow
import imcode.mps as mps
import imcode.dense as dense
import pytest
from .utils import seed_rng

@pytest.fixture(scope="module")
def dense_finite_short_time_real_hom():
    seed_rng("finite_short_time_real_hom")
    t=3
    J=np.random.normal()
    g=np.random.normal()
    h=np.random.normal()
    dt=dense.ising_T(t,J,g,h)
    ims=[dense.im_finite([dt]*L) for L in range(1,5)]
    return (ims,dt,(t,J,g,h))

@pytest.fixture(scope="module")
def dense_finite_short_time_complex_hom():
    seed_rng("finite_short_time_real_hom")
    t=3
    J=np.random.normal()+1.0j*np.random.normal()
    g=np.random.normal()+1.0j*np.random.normal()
    h=np.random.normal()+1.0j*np.random.normal()
    dt=dense.ising_T(t,J,g,h)
    ims=[dense.im_finite([dt]*L) for L in range(1,5)]
    return (ims,dt,(t,J,g,h))
@pytest.mark.skip("tbd later")
def test_sparse_finite_short_time_real_hom(dense_finite_short_time_real_hom):
    df=dense_finite_short_time_real_hom
    st=sparse.ising_T(*df[-1])
    for L in range(1,5):
        sim=sparse.im_finite([st]*L)
        assert sim==pytest.approx(df[0][L-1])
