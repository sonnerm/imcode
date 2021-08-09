import imcode.dense as dense
import numpy as np
import numpy.linalg as la
from functools import reduce
import pytest
def test_contract_2x3_mixed(seed_rng):
    L=2
    t=1
    init=np.random.normal(size=(4,4))+1.0j*np.random.normal(size=(4,4))
    final=np.random.normal(size=(4,4))+1.0j*np.random.normal(size=(4,4))
    gate=np.random.normal(size=(4,4))+1.0j*np.random.normal(size=(4,4))
    gate+=gate.T.conj()
    _,gate=la.eigh(gate)
    gate=np.eye(4)
    F=dense.brickwork.brickwork_F(L,[gate]*(L-1))
    acc=init
    for _ in range(t):
        acc=F@acc@F.T.conj()
    direct=np.trace(final@acc)
    bc=dense.brickwork.brickwork_La(t)
    transverse=bc@dense.brickwork.brickwork_Sb(t,dense.unitary_channel(gate),init,final)@bc
    assert transverse==pytest.approx(direct)

def test_contract_2x3_separable(seed_rng):
    L=2
    t=1
    init=[np.random.normal(size=(2,2))+1.0j*np.random.normal(size=(2,2)) for _ in range(L)]
    final=[np.random.normal(size=(2,2))+1.0j*np.random.normal(size=(2,2)) for _ in range(L)]
    gate=[np.random.normal(size=(2,2))+1.0j*np.random.normal(size=(2,2)) for _ in range(L)]
    F=dense.brickwork.brickwork_F(L,[dense.kron(gate)])
    acc=np.kron(init[0],init[1])
    for _ in range(t):
        acc=F@acc@F.T.conj()
    direct=np.trace(np.kron(final[0],final[1])@acc)
    bcl=dense.brickwork.brickwork_Lb(t,dense.unitary_channel(gate[1]),init[1],final[1])
    bcr=dense.brickwork.brickwork_Lb(t,dense.unitary_channel(gate[0]),init[0],final[0])
    transverse1=bcr@dense.brickwork.brickwork_Sa(t,dense.unitary_channel(np.eye(4)))@bcl
    assert transverse1==pytest.approx(direct)
    bc=dense.brickwork.brickwork_La(t)
    transverse=bc@dense.brickwork.brickwork_Sb(t,dense.unitary_channel(dense.kron(gate)),np.kron(init[0],init[1]),np.kron(final[0],final[1]))@bc
    assert transverse==pytest.approx(direct)

#
# def test_contract_3x3_mixed(seed_rng):
#     L=3
#     t=3
#     init=[np.random.normal(size=(2,2))+1.0j*np.random.normal((2,2)) for _ in range(L)]
#     final=[np.random.normal(size=(2,2))+1.0j*np.random.normal((2,2)) for _ in range(L)]
#     Js=np.random.normal(size=(L-1,))
#     gs=np.random.normal(size=(L,))
#     hs=np.random.normal(size=(L,))
#     Ts=[dense.ising.ising_T(t,J,g,h,i,f) for J,g,h,i,f in zip(Js,gs[:-1],hs[:-1],init[:-1],final[:-1])]
#     Ts.append(dense.ising.ising_g(t,gs[-1],init[-1],final[-1])@dense.ising.ising_h(t,hs[-1]))
#     F=dense.ising.ising_F(L,Js,gs,hs)
#     bc=dense.ising.open_boundary_im(t)
#     initv=dense.kron(init)
#     finalv=dense.kron(final)
#     for T in Ts:
#         bc=T@bc
#     transverse=dense.ising.open_boundary_im(t)@bc
#     for _ in range(t):
#         initv=F@initv@F.T.conj()
#     direct=np.trace(finalv@initv)
#     assert transverse==pytest.approx(direct)
#
# def test_contract_unity(seed_rng):
#     L=3
#     t=3
#     init=[np.random.normal(size=(2,2))+1.0j*np.random.normal((2,2)) for _ in range(L)]
#     init=[i@i.T.conj() for i in init]
#     init=[(i)/np.trace(i) for i in init]
#     final=[np.eye(2) for _ in range(L)]
#     Js=np.random.normal(size=(L,))
#     gs=np.random.normal(size=(L,))
#     hs=np.random.normal(size=(L,))
#     Ts=[dense.ising.ising_T(t,J,g,h,i,f) for J,g,h,i,f in zip(Js,gs[:-1],hs[:-1],init[:-1],final[:-1])]
#     Ts.append(dense.ising.ising_g(t,gs[-1],init[-1],final[-1])@dense.ising.ising_h(t,hs[-1]))
#     bc=dense.ising.open_boundary_im(t)
#     for T in Ts:
#         bc=T@bc
#     transverse=dense.ising.open_boundary_im(t)@bc
#     assert transverse==pytest.approx(1.0)
