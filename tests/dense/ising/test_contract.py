import imcode.dense as dense
import numpy as np
import pytest

def test_contract_3x3_mixed(seed_rng):
    L=3
    t=3
    init=[np.random.normal(size=(2,2)) for _ in range(L)]
    final=[np.random.normal(size=(2,2)) for _ in range(L)]
    Js=np.random.normal(size=(L-1,))
    gs=np.random.normal(size=(L,))
    hs=np.random.normal(size=(L,))
    Ts=[dense.ising.ising_T(t,J,g,h,i,f) for J,g,h,i,f in zip(Js,gs[:-1],hs[:-1],init[:-1],final[:-1])]
    Ts.append(dense.ising.ising_W(t,gs[-1],init[-1],final[-1])@dense.ising.ising_h(t,hs[-1]))
    F=dense.ising.ising_F(L,Js,gs,hs)
    bc=dense.ising.open_boundary_im(t)
    initv=dense.kron(init)
    finalv=dense.kron(final)
    for T in Ts:
        bc=T@bc
    transverse=dense.ising.open_boundary_im(t)@bc
    for _ in range(t):
        initv=F@initv@F.T.conj()
    direct=np.trace(finalv@initv)
    assert transverse==pytest.approx(direct)


def test_contract_3x3_pure(seed_rng):
    pass

def test_contract_3x3_boundary(seed_rng):
    pass
