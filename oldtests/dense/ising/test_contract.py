import imcode.dense as dense
import numpy as np
from functools import reduce
import pytest
def test_contract_1x3_mixed(seed_rng):
    L=1
    t=3
    init=np.random.normal(size=(2,2))+1.0j*np.random.normal(size=(2,2))
    final=np.random.normal(size=(2,2))+1.0j*np.random.normal(size=(2,2))
    g=np.random.normal()
    h=np.random.normal()
    F=dense.ising.ising_F(L,[0.0],[g],[h])
    acc=init
    for _ in range(t):
        acc=F@acc@F.T.conj()
    direct=np.trace(final@acc)
    bc=dense.ising.open_boundary_im(t)
    Wh=dense.ising.ising_g(t,g,init,final)@dense.ising.ising_h(t,h)
    transverse=bc@Wh@bc
    assert transverse==pytest.approx(direct)
    hW=dense.ising.ising_h(t,h)@dense.ising.ising_g(t,g,init,final)
    transverse=bc@hW@bc
    assert transverse==pytest.approx(direct)

def test_contract_3x3_mixed(seed_rng):
    L=3
    t=3
    init=[np.random.normal(size=(2,2))+1.0j*np.random.normal(size=(2,2)) for _ in range(L)]
    final=[np.random.normal(size=(2,2))+1.0j*np.random.normal(size=(2,2)) for _ in range(L)]
    Js=np.random.normal(size=(L-1,))
    gs=np.random.normal(size=(L,))
    hs=np.random.normal(size=(L,))
    Ts=[dense.ising.ising_T(t,J,g,h,i,f) for J,g,h,i,f in zip(Js,gs[:-1],hs[:-1],init[:-1],final[:-1])]
    Ts.append(dense.ising.ising_g(t,gs[-1],init[-1],final[-1])@dense.ising.ising_h(t,hs[-1]))
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

def test_contract_unity(seed_rng):
    L=3
    t=3
    init=[np.random.normal(size=(2,2))+1.0j*np.random.normal(size=(2,2)) for _ in range(L)]
    init=[i@i.T.conj() for i in init]
    init=[(i)/np.trace(i) for i in init]
    final=[np.eye(2) for _ in range(L)]
    Js=np.random.normal(size=(L,))
    gs=np.random.normal(size=(L,))
    hs=np.random.normal(size=(L,))
    Ts=[dense.ising.ising_T(t,J,g,h,i,f) for J,g,h,i,f in zip(Js,gs[:-1],hs[:-1],init[:-1],final[:-1])]
    Ts.append(dense.ising.ising_g(t,gs[-1],init[-1],final[-1])@dense.ising.ising_h(t,hs[-1]))
    bc=dense.ising.open_boundary_im(t)
    for T in Ts:
        bc=T@bc
    transverse=dense.ising.open_boundary_im(t)@bc
    assert transverse==pytest.approx(1.0)
