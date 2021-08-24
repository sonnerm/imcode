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
    # gate_even=np.eye(4)
    # gate_odd=np.eye(4)
    final=np.eye(4)
    Sa=dense.brickwork.brickwork_Sa(t,dense.unitary_channel(gate_even))
    Sb=dense.brickwork.brickwork_Sb(t,dense.unitary_channel(gate_odd),init,final)
    ims=[im for im in dense.brickwork.im_rectangle(Sa,Sb)]
    assert ims[-1]==pytest.approx(dense.ising.im_diag(Sa@Sb)) #correct ev
    assert ims[-1]==pytest.approx(ims[-2]) #convergence achieved
    assert ims[-1]!=pytest.approx(ims[1]) #but not immediately

def test_im_diamond_hom(seed_rng):
    gate=np.random.normal(size=(4,4))+1.0j*np.random.normal(size=(4,4))
    # gate=np.eye(4)
    gate=la.eigh(gate+gate.T.conj())[1]
    Sas=[dense.brickwork.brickwork_Sa(t,dense.unitary_channel(gate)) for t in range(1,3)]
    ims=[im for im in dense.brickwork.im_diamond(Sas)]
    for im,Sa in zip(ims,Sas):
        assert im ==pytest.approx(dense.brickwork.im_diag(Sa))

# def test_im_diamond_het(seed_rng):
#     t=5
#     Js,gs,hs=np.random.normal(size=(3,t))
#     Ts=[dense.ising.ising_T(t,J,g,h) for t,J,g,h in zip(range(1,t+1),Js,gs,hs)]
#     ims=[im for im in dense.ising.im_triangle(Ts)]
#     for im,t in zip(ims,range(1,t+1)):
#         Tsr=[dense.ising.ising_T(t,J,g,h) for J,g,h in zip(Js[:t],gs[:t],hs[:t])]
#         assert im ==pytest.approx(list(dense.ising.im_rectangle(Tsr))[-1])
# def test_im_triangle_hom(seed_rng):
#     J,g,h=np.random.normal(size=3)
#     t=5
#     init=np.random.normal(size=(2,2))+1.0j*np.random.normal(size=(2,2))
#     init=init.T.conj()@init
#     init/=np.trace(init)
#     Ts=[dense.ising.ising_T(t,J,g,h,init) for t in range(1,6,1)]
#     ims=[im for im in dense.ising.im_triangle(Ts)]
#     for im,T in zip(ims,Ts):
#         assert im ==pytest.approx(dense.ising.im_diag(T))
#
# def test_im_triangle_het(seed_rng):
#     t=5
#     Js,gs,hs=np.random.normal(size=(3,t))
#     inits=[np.random.normal(size=(2,2))+1.0j*np.random.normal(size=(2,2)) for i in range(t)]
#     inits=[i.T.conj()@i for i in inits]
#     inits=[i/np.trace(i) for i in inits]
#     Ts=[dense.ising.ising_T(t,J,g,h,i) for t,J,g,h,i in zip(range(1,t+1),Js,gs,hs,inits)]
#     ims=[im for im in dense.ising.im_triangle(Ts)]
#     for im,t in zip(ims,range(1,t+1)):
#         Tsr=[dense.ising.ising_T(t,J,g,h,i) for J,g,h,i in zip(Js[:t],gs[:t],hs[:t],inits[:t])]
#         assert im ==pytest.approx(list(dense.ising.im_rectangle(Tsr))[-1])
