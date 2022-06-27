import imcode.mps as mps
import numpy as np
import numpy.linalg as la
import imcode.dense as dense
import pytest
def test_im_rectangle_dense(seed_rng):
    t=3
    gates=[np.random.normal(size=(4,4))+1.0j*np.random.normal(size=(4,4)) for _ in range(t)]
    inits=[np.random.normal(size=(4,4))+1.0j*np.random.normal(size=(4,4)) for _ in range(t)]
    finals=[np.random.normal(size=(4,4))+1.0j*np.random.normal(size=(4,4)) for _ in range(t)]
    gates=[dense.unitary_channel(g) for g in gates]
    Sam=[mps.brickwork.brickwork_Sa(t,g) for g in gates[::2]]
    Sbm=[mps.brickwork.brickwork_Sb(t,g,i,f) for g,i,f in zip(gates[1::2],inits,finals)]
    Sad=[dense.brickwork.brickwork_Sa(t,g) for g in gates[::2]]
    Sbd=[dense.brickwork.brickwork_Sb(t,g,i,f) for g,i,f in zip(gates[1::2],inits,finals)]
    imsm=[im for im in mps.brickwork.im_rectangle(Sam,Sbm)]
    imsd=[im for im in dense.brickwork.im_rectangle(Sad,Sbd)]
    for imm,imd in zip(imsm,imsd):
        assert imm.to_dense() == pytest.approx(imd)

# def test_im_diamond_dense(seed_rng):
#     t=3
#     gates=[np.random.normal(size=(4,4))+1.0j*np.random.normal(size=(4,4)) for _ in range(t)]
#     gates=[dense.unitary_channel(g) for g in gates]
#     Sam=[mps.brickwork.brickwork_Sa(t,g) for g in gates]
#     Sad=[dense.brickwork.brickwork_Sa(t,g) for g in gates]
#     imsm=[im for im in mps.brickwork.im_diamond(Tsm)]
#     imsd=[im for im in dense.brickwork.im_diamond(Tsd)]
#     for imm,imd in zip(imsm,imsd):
#         assert imm.to_dense() == pytest.approx(imd)
#
def test_im_triangle_dense(seed_rng):
    t=3
    gates=[np.random.normal(size=(4,4))+1.0j*np.random.normal(size=(4,4)) for _ in range(t)]
    gates=[dense.unitary_channel(g) for g in gates]
    inits=[np.random.normal(size=(4,4))+1.0j*np.random.normal(size=(4,4)) for i in range(t)]
    finals=[np.random.normal(size=(4,4))+1.0j*np.random.normal(size=(4,4)) for i in range(t)]
    Sam=[mps.brickwork.brickwork_Sa(t,g) for t,g in zip(range(1,t+1),gates[::2])]
    Sbm=[mps.brickwork.brickwork_Sb(t,g,i,f) for t,g,i,f in zip(range(1,t+1),gates[1::2],inits,finals)]
    Sad=[dense.brickwork.brickwork_Sa(t,g) for t,g in zip(range(1,t+1),gates[::2])]
    Sbd=[dense.brickwork.brickwork_Sb(t,g,i,f) for t,g,i,f in zip(range(1,t+1),gates[1::2],inits,finals)]
    imsm=[im for im in mps.brickwork.im_triangle(Sam,Sbm)]
    imsd=[im for im in dense.brickwork.im_triangle(Sad,Sbd)]
    for imm,imd in zip(imsm,imsd):
        assert imm.to_dense()==pytest.approx(imd)

def test_im_rectangle(seed_rng):
    gatea=np.random.normal(size=(4,4))+1.0j*np.random.normal(size=(4,4))
    gatea=la.eigh(gatea+gatea.T.conj())[1]
    gateb=np.random.normal(size=(4,4))+1.0j*np.random.normal(size=(4,4))
    gateb=la.eigh(gateb+gateb.T.conj())[1]
    init=np.random.normal(size=(4,4))+1.0j*np.random.normal(size=(4,4))
    init=init.T.conj()@init
    init/=np.trace(init)
    final=np.random.normal(size=(4,4))+1.0j*np.random.normal(size=(4,4))
    final=final.T.conj()@final
    final/=np.trace(final)
    t=5
    Sam=mps.brickwork.brickwork_Sa(t,dense.unitary_channel(gatea))
    Sbm=mps.brickwork.brickwork_Sb(t,dense.unitary_channel(gateb),init,final)
    ims=[im for im in mps.brickwork.im_rectangle(Sam,Sbm,chi_max=16)]
    # print(ims[-1].tpmps.chi)
    a=ims[-1].conj()@ims[-2]
    b=np.sqrt((ims[-1].conj()@ims[-1])*(ims[-2].conj()@ims[-2]))
    assert a==pytest.approx(b,rel=1e-2) #convergence achieved, bond dimension seems to be a major issue
def test_im_triangle_het(seed_rng):
    t=5
    gates=[np.random.normal(size=(4,4))+1.0j*np.random.normal(size=(4,4)) for _ in range(t)]
    gates=[la.eigh(g.T.conj()+g)[1] for g in gates]
    gates=[dense.unitary_channel(g) for g in gates]
    inits=[np.random.normal(size=(4,4))+1.0j*np.random.normal(size=(4,4)) for i in range(t)]
    inits=[i.T.conj()@i for i in inits]
    inits=[i/np.trace(i) for i in inits]
    Sas=[mps.brickwork.brickwork_Sa(t,g) for t,g in zip(range(1,t+1),gates[::2])]
    Sbs=[mps.brickwork.brickwork_Sb(t,g,i) for t,g,i in zip(range(1,t+1),gates[1::2],inits)]
    ims=[im for im in mps.brickwork.im_triangle(Sas,Sbs,chi_max=16)]
    for im,t in zip(ims,range(1,t+1)):
        Sar=[mps.brickwork.brickwork_Sa(t,g) for g in gates[:2*t+1:2]]
        Sbr=[mps.brickwork.brickwork_Sb(t,g,i) for g,i in zip(gates[1:2*t+1:2],inits)]
        imo=list(mps.brickwork.im_rectangle(Sar,Sbr,chi_max=16))[-1]
        a=im.conj()@imo
        b=np.sqrt((im.conj()@im)*(imo.conj()@imo))
        assert a==pytest.approx(b,rel=1e-2) #convergence achieved
# def test_im_diamond_het(seed_rng):
#     t=10
#     gates=[np.random.normal(size=(4,4))+1.0j*np.random.normal(size=(4,4)) for _ in range(t)]
#     gates=[dense.unitary_channel(g) for g in gates]
#     Sas=[mps.brickwork.brickwork_Sa(t,g) for t,g in zip(range(1,t+1,2),gates[::2])]
#     Sbs=[mps.brickwork.brickwork_Sb(t,g,i) for t,g,i in zip(range(2,t+1,2),gates[1::2])]
#     ims=[im for im in mps.brickwork.im_diamond(Ts,chi_max=64)]
#     for im,t in zip(ims,range(1,t+1)):
#        # Tsr=[mps.brickwork.ising_T(t,J,g,h) for J,g,h in zip(Js[:t],gs[:t],hs[:t])]
#         imo=list(mps.ising.im_rectangle(Tsr,chi_max=64))[-1]
#         assert im.conj()@imo==pytest.approx(np.sqrt((im.conj()@im)*(imo.conj()@imo))) #convergence achieved
