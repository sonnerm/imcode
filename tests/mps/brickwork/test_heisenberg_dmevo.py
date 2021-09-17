import imcode.mps as mps
import imcode.dense as dense
import scipy.linalg as la
import numpy as np
import pytest

def test_boundary_single_dmevo_heisenberg_bw(seed_rng):
    L=4
    t=10
    chi=64
    Jx,Jy,Jz=np.random.normal(size=(3,L))
    Jx,Jy,Jz=([Jx[0]]*L,)*3
    hx,hy,hz=np.random.normal(size=(3,L+1))
    hx,hy,hz=np.zeros((3,L+1))
    sagates=[dense.brickwork.heisenberg_gate(jx,jy,jz) for jx,jy,jz in zip(Jx[1::2],Jy[1::2],Jz[1::2])]
    sbgates=[dense.brickwork.heisenberg_gate(jx,jy,jz,hx,hy,hz,hxe,hye,hze) for jx,jy,jz,hx,hy,hz,hxe,hye,hze in zip(Jx[::2],Jy[::2],Jz[::2],hx[::2],hy[::2],hz[::2],hx[1::2],hy[1::2],hz[1::2])]
    Sas=[mps.brickwork.brickwork_Sa(t,dense.unitary_channel(g)) for g in sagates]
    Sbs=[mps.brickwork.brickwork_Sb(t,dense.unitary_channel(g)) for g in sbgates]
    im=list(mps.brickwork.im_rectangle(Sas,Sbs,chi_max=chi))[-1]
    init=np.random.normal(size=(2,2))+np.random.normal(size=(2,2))*1.0j
    init=init+init.T.conj()
    init=init@init
    init/=np.trace(init)
    # init=np.array([[1,0],[0,0]])#TODO: remove
    lop=dense.brickwork.heisenberg_F(1,[],[],[],[hx[-1]],[hy[-1]],[hz[-1]])
    dms=mps.brickwork.boundary_dm_evolution(im,dense.unitary_channel(lop),init)
    F=dense.brickwork.heisenberg_F(L+1,Jx,Jy,Jz,hx,hy,hz)
    state=dense.kron([np.eye(2)/2]*(L)+[init])
    summi=dense.kron([np.eye(2)]*(L))
    ddms=[np.einsum("ab,acbd->cd",summi,state.reshape((2**L),2,(2**L),2)).reshape((2,2))]
    for i in range(t):
        state=F@state@F.T.conj()
        ddms.append(np.einsum("ab,acbd->cd",summi,state.reshape((2**L),2,(2**L),2)).reshape((2,2)))
    for d,dd in zip(dms[::2],ddms):
        assert d==pytest.approx(dd)
@pytest.mark.skip()
def test_boundary_double_dmevo_brickwork(seed_rng):
    L=4
    t=10
    chi=256
    gates=[np.random.normal(size=(4,4))+1.0j*np.random.normal(size=(4,4))*1.0j for _ in range(L+1)]
    gates=[la.eigh(g+g.T.conj())[1] for g in gates]
    Sas=[mps.brickwork.brickwork_Sa(t,dense.unitary_channel(g)) for g in gates[1:-1:2]]
    Sbs=[mps.brickwork.brickwork_Sb(t,dense.unitary_channel(g)) for g in gates[:-1:2]]
    init=np.random.normal(size=(4,4))+np.random.normal(size=(4,4))*1.0j
    init=init+init.T.conj()
    init=init@init
    init/=np.trace(init)
    im=list(mps.brickwork.im_rectangle(Sas,Sbs,chi_max=chi))[-1]
    dms=mps.brickwork.boundary_dm_evolution(im,dense.unitary_channel(gates[-1].reshape((2,2,2,2)).transpose([1,0,3,2]).reshape((4,4))),init)
    F=dense.brickwork.brickwork_F(L+2,[g.reshape((2,2,2,2)).transpose([1,0,3,2]).reshape((4,4)) for g in gates])
    state=dense.kron([np.eye(2)/2]*(L)+[init])
    summi=dense.kron([np.eye(2)]*(L))
    ddms=[np.einsum("ab,acbd->cd",summi,state.reshape((2**L),4,(2**L),4)).reshape((4,4))]
    for i in range(t):
        state=F@state@F.T.conj()
        ddms.append(np.einsum("ab,acbd->cd",summi,state.reshape((2**L),4,(2**L),4)).reshape((4,4)))
    for d,dd in zip(dms[::2],ddms):
        assert d==pytest.approx(dd)
@pytest.mark.skip()
def test_embedded_double_dmevo_brickwork(seed_rng):
    Ll=2
    Lr=2
    t=10
    chi=256
    gates=[np.random.normal(size=(4,4))+1.0j*np.random.normal(size=(4,4))*1.0j for _ in range(Ll+Lr+1)]
    gates=[la.eigh(g+g.T.conj())[1] for g in gates]
    gates=[g.reshape((2,2,2,2)).transpose([1,0,3,2]).reshape((4,4))@g for g in gates]
    Sasl=[mps.brickwork.brickwork_Sa(t,dense.unitary_channel(g)) for g in gates[1:Ll:2]]
    Sbsl=[mps.brickwork.brickwork_Sb(t,dense.unitary_channel(g)) for g in gates[:Ll:2]]
    Sasr=[mps.brickwork.brickwork_Sa(t,dense.unitary_channel(g.reshape((2,2,2,2)).transpose([1,0,3,2]).reshape((4,4)))) for g in gates[Ll+2::2]]
    Sbsr=[mps.brickwork.brickwork_Sb(t,dense.unitary_channel(g.reshape((2,2,2,2)).transpose([1,0,3,2]).reshape((4,4)))) for g in gates[Lr+1::2]]

    iml=list(mps.brickwork.im_rectangle(Sasl,Sbsl,chi_max=chi))[-1]
    imr=list(mps.brickwork.im_rectangle(Sasr,Sbsr,chi_max=chi))[-1]

    init=np.random.normal(size=(4,4))+np.random.normal(size=(4,4))*1.0j
    init=init+init.T.conj()
    init=init@init
    init/=np.trace(init)
    dms=mps.brickwork.embedded_dm_evolution(iml,dense.unitary_channel(gates[Ll]),imr,init)

    F=dense.brickwork.brickwork_F(Ll+Lr+2,gates)
    state=dense.kron([np.eye(2)/2]*(Ll)+[init]+[np.eye(2)/2]*(Lr))
    summil=dense.kron([np.eye(2)]*(Ll))
    summir=dense.kron([np.eye(2)]*(Lr))
    ddms=[np.einsum("ad,abcdef,cf->be",summil,state.reshape((2**Ll),4,(2**Lr),(2**Ll),4,(2**Lr)),summir)]
    for i in range(t):
        state=F@state@F.T.conj()
        ddms.append(np.einsum("ad,abcdef,cf->be",summil,state.reshape((2**Ll),4,(2**Lr),(2**Ll),4,(2**Lr)),summir))
    for d,dd in zip(dms[::2],ddms):
        assert d==pytest.approx(dd)
