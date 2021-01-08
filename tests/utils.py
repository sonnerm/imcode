from imcode import sparse
import pytest
import numpy as np
def sparse_eq(sp,de):
    print(sparse.sparse_to_dense(sp))
    print(de)

    assert sparse.sparse_to_dense(sp)==pytest.approx(de)
    assert sparse.sparse_to_dense(sp.T)==pytest.approx(de.T)
    assert sparse.sparse_to_dense(sp.conj())==pytest.approx(de.conj())
    assert sparse.sparse_to_dense(sp.T.conj())==pytest.approx(de.T.conj())
