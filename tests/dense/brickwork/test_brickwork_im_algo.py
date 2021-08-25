import imcode.dense as dense
import numpy as np
import numpy.linalg as la
from functools import reduce
import pytest

def test_im_rectangle(seed_rng):
    t=2
    gate_even=np.random.normal(size=(4,4))+1.0j*np.random.normal(size=(4,4))
    gate_odd=np.random.normal(size=(4,4))+1.0j*np.random.normal(size=(4,4))
    gate_even=la.eigh(gate_even+gate_even.T.conj())[1]
    gate_odd=la.eigh(gate_odd+gate_odd.T.conj())[1]
    init=np.random.normal(size=(4,4))+1.0j*np.random.normal(size=(4,4))
    init=init@init.T.conj()
    init/=np.trace(init)
    init=np.eye(4)/4
    final=np.eye(4)
    Sa=dense.brickwork.brickwork_Sa(t,dense.unitary_channel(gate_even))
    Sb=dense.brickwork.brickwork_Sb(t,dense.unitary_channel(gate_odd),init,final)
    ims=[im for im in dense.brickwork.im_rectangle(Sa,Sb)]
    assert ims[-1]==pytest.approx(ims[-2]) #convergence achieved
    # assert ims[-1]!=pytest.approx(ims[1]) #but not immediately, well at t=2 ...
    assert ims[-1]==pytest.approx(dense.brickwork.im_diag(Sa,Sb),abs=1e-8,rel=1e-10) #correct ev

def test_im_diamond_hom(seed_rng):
    gate=np.random.normal(size=(4,4))+1.0j*np.random.normal(size=(4,4))
    gate=la.eigh(gate+gate.T.conj())[1]
    Sas=[dense.brickwork.brickwork_Sa(t,dense.unitary_channel(gate)) for t in range(1,3)]
    Sbs=[dense.brickwork.brickwork_Sb(t,dense.unitary_channel(gate)) for t in range(1,3)]
    ims=[im for im in dense.brickwork.im_diamond(Sas)]
    for im,Sa,Sb in zip(ims,Sas,Sbs):
        assert im ==pytest.approx(dense.brickwork.im_diag(Sa,Sb),abs=1e-8,rel=1e-10)


def test_im_diamond_het(seed_rng):
    t=3
    gates=[np.random.normal(size=(4,4))+1.0j*np.random.normal(size=(4,4)) for _ in range(t)]
    gates=[la.eigh(g+g.T.conj())[1] for g in gates]
    Sas=[dense.brickwork.brickwork_Sa(t,dense.unitary_channel(g)) for t,g in zip(range(1,t+1),gates)]
    ims=[im for im in dense.brickwork.im_diamond(Sas)]
    for im,t in zip(ims,range(1,t+1)):
        if t%2==1:
            continue #im rectangle only works for even t
        Sbsr=[dense.brickwork.brickwork_Sb(t,dense.unitary_channel(g)) for g in gates[:t:2]]
        Sasr=[dense.brickwork.brickwork_Sa(t,dense.unitary_channel(g)) for g in gates[1:t:2]]
        assert im ==pytest.approx(list(dense.brickwork.im_rectangle(Sasr,Sbsr))[-1],abs=1e-8,rel=1e-10)
# def test_im_triangle_hom(seed_rng):
#     gate=np.random.normal(size=(4,4))+1.0j*np.random.normal(size=(4,4))
#     init=np.random.normal(size=(4,4))+1.0j*np.random.normal(size=(4,4))
#     init=init.T.conj()@init
#     init/=np.trace(init)
#     final=np.eye(4)
#     gate=la.eigh(gate+gate.T.conj())[1]
#     Sas=[dense.brickwork.brickwork_Sa(t,dense.unitary_channel(gate)) for t in range(1,3)]
#     Sbs=[dense.brickwork.brickwork_Sb(t,dense.unitary_channel(gate),init,final) for t in range(1,3)]
#     ims=[im for im in dense.brickwork.im_triangle(Sas,Sbs)]
#     for im,Sa,Sb in zip(ims,Sas,Sbs):
#         assert im ==pytest.approx(dense.brickwork.im_diag(Sa,Sb),abs=1e-8,rel=1e-10)
# def test_im_triangle_het(seed_rng):
#     t=2
#     gates=[np.random.normal(size=(4,4))+1.0j*np.random.normal(size=(4,4)) for _ in range(t)]
#     inits=[np.random.normal(size=(4,4))+1.0j*np.random.normal(size=(4,4)) for _ in range(t//2)]
#     inits=[i@i.T.conj() for i in inits]
#     inits=[i/np.trace(i) for i in inits]
#     inits=[np.eye(4)/4 for _ in range(t//2)]
#     finals=[np.eye(4) for _ in range(t//2)]
#     gates=[la.eigh(g+g.T.conj())[1] for g in gates]
#     Sas=[dense.brickwork.brickwork_Sa(t,dense.unitary_channel(g)) for t,g in zip(range(1,t+1),gates[1::2])]
#     Sbs=[dense.brickwork.brickwork_Sb(t,dense.unitary_channel(g),i,f) for t,g,i,f in zip(range(1,t+1),gates[::2],inits,finals)]
#     ims=[im for im in dense.brickwork.im_triangle(Sas,Sbs)]
#     for im,t in zip(ims,range(2,t+1,2)):
#         Sbsr=[dense.brickwork.brickwork_Sb(t,dense.unitary_channel(g),i,f) for g,i,f in zip(gates[:t:2],inits,finals)]
#         Sasr=[dense.brickwork.brickwork_Sa(t,dense.unitary_channel(g)) for g in gates[1:t:2]]
#         assert im ==pytest.approx(list(dense.brickwork.im_rectangle(Sasr,Sbsr))[-1],abs=1e-8,rel=1e-10)
