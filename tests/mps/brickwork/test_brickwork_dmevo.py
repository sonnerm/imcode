import imcode.mps as mps
import imcode.dense as dense
import scipy.linalg as la
import numpy as np
import pytest


def test_boundary_single_dmevo_brickwork(seed_rng):
    L=4
    t=10
    chi=256
    gates=[np.random.normal(size=(4,4))+1.0j*np.random.normal(size=(4,4))*1.0j for _ in range(L)]
    gates=[la.eigh(g+g.T.conj())[1] for g in gates]
    lop=np.random.normal(size=(2,2))+np.random.normal(size=(2,2))*1.0j
    lop=la.eigh(lop+lop.T.conj())[1]
    Sas=[mps.brickwork.brickwork_Sa(t,dense.unitary_channel(g)) for g in gates[1::2]]
    Sbs=[mps.brickwork.brickwork_Sb(t,dense.unitary_channel(g)) for g in gates[::2]]
    im=list(mps.brickwork.im_rectangle(Sas,Sbs,chi_max=chi))[-1]
    init=np.random.normal(size=(2,2))+np.random.normal(size=(2,2))*1.0j
    init=init+init.T.conj()
    init=init@init
    init/=np.trace(init)
    # init=np.array([[1,0],[0,0]])#TODO: remove
    dms=mps.brickwork.boundary_dm_evolution(im,dense.unitary_channel(lop),init)
    F=dense.brickwork.brickwork_F(L+1,[g.reshape((2,2,2,2)).transpose([1,0,3,2]).reshape((4,4)) for g in gates])
    F=F@np.kron(np.eye(2**L),lop)
    state=dense.kron([np.eye(2)/2]*(L)+[init])
    summi=dense.kron([np.eye(2)]*(L))
    ddms=[np.einsum("ab,acbd->cd",summi,state.reshape((2**L),2,(2**L),2)).reshape((2,2))]
    for i in range(t):
        state=F@state@F.T.conj()
        ddms.append(np.einsum("ab,acbd->cd",summi,state.reshape((2**L),2,(2**L),2)).reshape((2,2)))
    for d,dd in zip(dms[::2],ddms):
        assert d==pytest.approx(dd)

# def test_boundary_double_dmevo_brickwork(seed_rng):
#     L=4
#     t=10
#     chi=32
#     Js=np.random.normal(size=(L,))
#     gs=np.random.normal(size=(L+1,))
#     hs=np.random.normal(size=(L+1,))
#     Ts=[mps.ising.ising_T(t,J,g,h,np.eye(2)/2,np.eye(2)) for J,g,h in zip(Js,gs[:-1],hs[:-1])]
#     lop=la.expm(1.0j*hs[-1]*dense.SZ)@la.expm(1.0j*gs[-1]*dense.SX)
#     init=np.random.normal(size=(2,2))+np.random.normal(size=(2,2))*1.0j
#     init=init+init.T.conj()
#     init=init@init
#     init/=np.trace(init)
#     im=mps.ising.open_boundary_im(t)
#     for T in Ts:
#         im=(T@im).contract(chi_max=chi)
#     dms=mps.ising.boundary_dm_evolution(im,dense.unitary_channel(lop),init)
#     F=dense.ising.ising_F(L+1,Js,gs,hs)
#     # Fc=dense.unitary_channel(F)
#     # state=dense.outer([np.eye(2).ravel()/2]*(L)+[init.ravel()])
#     state=dense.kron([np.eye(2)/2]*(L)+[init])
#     summi=dense.kron([np.eye(2)]*(L))
#     ddms=[np.einsum("ab,acbd->cd",summi,state.reshape((2**L),2,(2**L),2)).reshape((2,2))]
#     for i in range(t):
#         state=F@state@F.T.conj()
#         ddms.append(np.einsum("ab,acbd->cd",summi,state.reshape((2**L),2,(2**L),2)).reshape((2,2)))
#     for d,dd in zip(dms[::2],ddms):
#         assert d==pytest.approx(dd)
#
# def test_embedded_double_dmevo_brickwork(seed_rng):
#     Ll=2
#     Lr=3
#     t=10
#     chi=32
#     Js=np.random.normal(size=(Ll+Lr+1,))
#     gs=np.random.normal(size=(Ll+Lr+2,))
#     hs=np.random.normal(size=(Ll+Lr+2,))
#     Tsl=[mps.ising.ising_T(t,J,g,h) for J,g,h in zip(Js[:Ll],gs[:Ll],hs[:Ll])]
#     Tsr=[mps.ising.ising_T(t,J,g,h) for J,g,h in zip(Js[-1:-Lr-1:-1],gs[-1:-Lr-1:-1],hs[-1:-Lr-1:-1])]
#
#     lop=dense.ising.ising_F(2,Js[Ll:Ll+1],gs[Ll:Ll+2],hs[Ll:Ll+2])
#     init=np.random.normal(size=(4,4))+np.random.normal(size=(4,4))*1.0j
#     init=init+init.T.conj()
#     init=init@init
#     init/=np.trace(init)
#     lim=mps.ising.open_boundary_im(t)
#     for T in Tsl:
#         lim=(T@lim).contract(chi_max=chi)
#     assert (np.array(lim.tpmps.chi)<=chi).all()
#     rim=mps.ising.open_boundary_im(t)
#     for T in Tsr:
#         rim=(T@rim).contract(chi_max=chi)
#     assert (np.array(rim.tpmps.chi)<=chi).all()
#     dms=mps.ising.embedded_dm_evolution(lim,dense.unitary_channel(lop),rim,init)
#     F=dense.ising.ising_F(Ll+Lr+2,Js,gs,hs)
#     state=dense.kron([np.eye(2)/2]*(Ll)+[init]+[np.eye(2)/2]*(Lr))
#     summil=dense.kron([np.eye(2)]*(Ll))
#     summir=dense.kron([np.eye(2)]*(Lr))
#     ddms=[np.einsum("ad,abcdef,cf->be",summil,state.reshape((2**Ll),4,(2**Lr),(2**Ll),4,(2**Lr)),summir)]
#     for i in range(t):
#         state=F@state@F.T.conj()
#         ddms.append(np.einsum("ad,abcdef,cf->be",summil,state.reshape((2**Ll),4,(2**Lr),(2**Ll),4,(2**Lr)),summir))
#     for d,dd in zip(dms[::2],ddms):
#         assert d==pytest.approx(dd)
