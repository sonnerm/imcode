import imcode.mps as mps
import numpy as np
import pytest
def test_contract_6x6_mixed(seed_rng):
    L=6
    t=6
    chi=64
    init=[np.random.normal(size=(2,2))+1.0j*np.random.normal((2,2)) for _ in range(L)]
    final=[np.random.normal(size=(2,2))+1.0j*np.random.normal((2,2)) for _ in range(L)]
    init=[i.T.conj()+i for i in init]
    final=[f.T.conj()+f for f in final]
    Js=np.random.normal(size=(L-1,))
    gs=np.random.normal(size=(L,))
    hs=np.random.normal(size=(L,))
    Ts=[mps.ising.ising_T(t,J,g,h,i,f) for J,g,h,i,f in zip(Js,gs[:-1],hs[:-1],init[:-1],final[:-1])]
    Ts.append((mps.ising.ising_W(t,gs[-1],init[-1],final[-1])@mps.ising.ising_h(t,hs[-1])).contract())
    bc=mps.ising.open_boundary_im(t)
    for T in Ts:
        bc=(T@bc).contract(chi_max=chi)
    F=mps.ising.ising_F(L,Js,gs,hs)
    Fc=mps.unitary_channel(F)
    transverse=mps.ising.open_boundary_im(t)@bc
    initv=mps.MPS.from_product_state([i.ravel() for i in init])
    for _ in range(t):
        initv=(Fc@initv).contract(chi_max=chi)
    finalv=mps.MPS.from_product_state([f.T.ravel() for f in final])
    direct=finalv@initv
    assert transverse==pytest.approx(direct)

def test_contract_unity(seed_rng):
    L=6
    t=6
    chi=64
    init=[np.random.normal(size=(2,2))+1.0j*np.random.normal((2,2)) for _ in range(L)]
    init=[i@i.T.conj() for i in init]
    init=[(i)/np.trace(i) for i in init]
    # init=[np.eye(2)/2 for _ in range(L)]
    final=[np.eye(2) for _ in range(L)]
    Js=np.random.normal(size=(L-1,))
    gs=np.random.normal(size=(L,))
    hs=np.random.normal(size=(L,))
    Ts=[mps.ising.ising_T(t,J,g,h,i,f) for J,g,h,i,f in zip(Js,gs[:-1],hs[:-1],init[:-1],final[:-1])]
    Ts.append(mps.ising.ising_W(t,gs[-1],init[-1],final[-1])@mps.ising.ising_h(t,hs[-1]))
    bc=mps.ising.open_boundary_im(t)
    for T in Ts:
        bc=(T@bc).contract(chi_max=chi)
    transverse=mps.ising.open_boundary_im(t)@bc
    assert transverse==pytest.approx(1.0)
