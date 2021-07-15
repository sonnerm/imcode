import imcode.dense as dense
import pytest
import numpy as np
def test_unitary_channel(seed_rng):
    L=4
    F=np.random.normal(size=(2**L,2**L))+1.0j*np.random.normal(size=(2**L,2**L))
    init=[np.random.normal(size=(2,2))+1.0j*np.random.normal(size=(2,2)) for _ in range(L)]
    final=[np.random.normal(size=(2,2))+1.0j*np.random.normal(size=(2,2)) for _ in range(L)]
    initm=dense.kron(init)
    finalm=dense.kron(final)
    initv=dense.outer([i.ravel() for i in init])
    finalv=dense.outer([f.T.conj().ravel() for f in final])
    trad=np.trace(finalm@F@initm@F.T.conj())
    Fc=dense.unitary_channel(F)
    newm=finalv.conj()@Fc@initv
    assert trad==pytest.approx(newm)
#
def test_unitary_channel_generic(seed_rng):
    L=2
    F=np.random.normal(size=(2**L,2**L))+1.0j*np.random.normal(size=(2**L,2**L))
    init=np.random.normal(size=(2**L,2**L))+1.0j*np.random.normal(size=(2**L,2**L))
    trad=F@init@F.T.conj()
    Fc=dense.unitary_channel(F)
    newm=dense.state_to_operator(Fc@dense.operator_to_state(init))
    assert trad==pytest.approx(newm)
