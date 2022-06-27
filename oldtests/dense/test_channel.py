import imcode.dense as dense
import pytest
import numpy as np
def test_unitary_channel_sandwich(seed_rng):
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
    L=4
    F=np.random.normal(size=(2**L,2**L))+1.0j*np.random.normal(size=(2**L,2**L))
    init=np.random.normal(size=(2**L,2**L))+1.0j*np.random.normal(size=(2**L,2**L))
    trad=F@init@F.T.conj()
    Fc=dense.unitary_channel(F)
    newm=dense.state_to_operator(Fc@dense.operator_to_state(init))
    assert trad==pytest.approx(newm)
def test_dephasing_zz_channel():
    pd=dense.dephasing_channel(1.0)
    assert pd==pytest.approx(np.diag(dense.operator_to_state(np.eye(2))))
    imd=dense.dephasing_channel(0.8)
    assert imd==pytest.approx(np.diag(dense.operator_to_state(np.array([[1,0.2],[0.2,1]]))))
    nd=dense.dephasing_channel(0.0)
    assert nd==pytest.approx(np.eye(4))

def test_dephasing_xx_channel():
    basis=np.array([[1,1],[1,-1]])/np.sqrt(2)
    pd=dense.dephasing_channel(1.0,basis)
    imd=dense.dephasing_channel(0.4,basis)
    it=dense.operator_to_state(np.eye(2)/2)
    assert pd@it==pytest.approx(it)
    assert imd@it==pytest.approx(it)
    xp=dense.operator_to_state(np.ones((2,2)))
    assert imd@xp==pytest.approx(xp)
    assert pd@xp==pytest.approx(xp)
    zp=dense.operator_to_state(np.array([[1,0],[0,0]]))
    assert pd@zp==pytest.approx(it)
    assert imd@zp==pytest.approx(zp*0.6+it*0.4)
    nd=dense.dephasing_channel(0.0,basis)
    assert nd==pytest.approx(np.eye(4))

def test_depolarizing_it_channel():
    pd=dense.depolarizing_channel(1.0)
    assert pd==pytest.approx(np.array([[0.5,0,0,0.5],[0,0,0,0],[0,0,0,0],[0.5,0,0,0.5]]))
    imd=dense.depolarizing_channel(0.4)
    assert imd==pytest.approx(np.array([[0.8,0.0,0.0,0.2],[0.0,0.6,0.0,0.0],[0.0,0.0,0.6,0.0],[0.2,0.0,0.0,0.8]]))
    nd=dense.depolarizing_channel(0.0)
    assert nd==pytest.approx(np.eye(4))

def test_depolarizing_zp_channel():
    dm=np.array([[1,0],[0,0]])
    pd=dense.depolarizing_channel(1.0,dm)
    assert pd==pytest.approx(np.array([[1,0,0,1],[0,0,0,0],[0,0,0,0],[0,0,0,0]]))
    imd=dense.depolarizing_channel(0.4,dm)
    assert imd==pytest.approx(np.array([[1.0,0.0,0.0,0.4],[0.0,0.6,0.0,0.0],[0.0,0.0,0.6,0.0],[0.0,0.0,0.0,0.6]]))
    nd=dense.depolarizing_channel(0.0,dm)
    assert nd==pytest.approx(np.eye(4))
