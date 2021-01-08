from imcode import sparse
import pytest
import numpy as np
import hashlib
def sparse_eq(sp,de):
    assert sparse.sparse_to_dense(sp)==pytest.approx(de)
    assert sparse.sparse_to_dense(sp.T)==pytest.approx(de.T)
    assert sparse.sparse_to_dense(sp.adjoint())==pytest.approx(de.T.conj())
    assert sparse.sparse_to_dense(sp.T.adjoint())==pytest.approx(de.conj())
def seed_rng(stri):
    np.random.seed(int.from_bytes(hashlib.md5(stri.encode('utf-8')).digest(),"big")%2**32)
