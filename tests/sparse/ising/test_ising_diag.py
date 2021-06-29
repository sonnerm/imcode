import imcode.sparse as sparse
import imcode.dense as dense
import numpy as np
import pytest
def test_sparse_ising_diag(seed_rng):
    L=5
    J=np.random.normal(size=L)+1.0j*np.random.normal(size=L)
    h=np.random.normal(size=L)+1.0j*np.random.normal(size=L)
    sdi=sparse.ising.ising_diag(J,h)
    assert sdi.dtype==np.complex_
    assert np.diag(sdi)==pytest.approx(dense.ising.ising_H(J,[0.0]*L,h))
    J=np.random.normal(size=L)
    h=np.random.normal(size=L)
    sdi=sparse.ising.ising_diag(J,h)
    assert sdi.dtype==np.float_
    assert np.diag(sdi)==pytest.approx(dense.ising.ising_H(J,[0.0]*L,h))
