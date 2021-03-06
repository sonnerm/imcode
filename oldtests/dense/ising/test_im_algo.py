import numpy as np
import imcode.dense as dense
import pytest
def test_im_rectangle(seed_rng):
    J,g,h=np.random.normal(size=3)
    init=np.random.normal(size=(2,2))+1.0j*np.random.normal(size=(2,2))
    init=init@init.T.conj()
    init/=np.trace(init)
    t=5
    T=dense.ising.ising_T(t,J,g,h,init)
    ims=[im for im in dense.ising.im_rectangle(T)]
    assert ims[-1]==pytest.approx(ims[-2]) #convergence achieved
    assert ims[-1]!=pytest.approx(ims[1]) #but not immediately
    assert ims[-1]==pytest.approx(dense.ising.im_diag(T)) #correct ev

def test_im_diamond_hom(seed_rng):
    J,g,h=np.random.normal(size=3)
    t=5
    Ts=[dense.ising.ising_T(t,J,g,h) for t in range(1,6,2)]
    ims=[im for im in dense.ising.im_diamond(Ts)]
    for im,T in zip(ims,Ts):
        assert im ==pytest.approx(dense.ising.im_diag(T))

def test_im_diamond_het(seed_rng):
    t=5
    Js,gs,hs=np.random.normal(size=(3,t//2))
    Ts=[dense.ising.ising_T(t,J,g,h) for t,J,g,h in zip(range(1,6,2),Js,gs,hs)]
    ims=[im for im in dense.ising.im_diamond(Ts)]
    for im,t in zip(ims,range(1,t+1,2)):
        Tsr=[dense.ising.ising_T(t,J,g,h) for J,g,h in zip(Js[:t//2+1],gs[:t//2+1],hs[:t//2+1])]
        assert im ==pytest.approx(list(dense.ising.im_rectangle(Tsr))[-1])
def test_im_triangle_hom(seed_rng):
    J,g,h=np.random.normal(size=3)
    t=5
    init=np.random.normal(size=(2,2))+1.0j*np.random.normal(size=(2,2))
    init=init.T.conj()@init
    init/=np.trace(init)
    Ts=[dense.ising.ising_T(t,J,g,h,init) for t in range(1,6,1)]
    ims=[im for im in dense.ising.im_triangle(Ts)]
    for im,T in zip(ims,Ts):
        assert im ==pytest.approx(dense.ising.im_diag(T))

def test_im_triangle_het(seed_rng):
    t=5
    Js,gs,hs=np.random.normal(size=(3,t))
    inits=[np.random.normal(size=(2,2))+1.0j*np.random.normal(size=(2,2)) for i in range(t)]
    inits=[i.T.conj()@i for i in inits]
    inits=[i/np.trace(i) for i in inits]
    Ts=[dense.ising.ising_T(t,J,g,h,i) for t,J,g,h,i in zip(range(1,t+1),Js,gs,hs,inits)]
    ims=[im for im in dense.ising.im_triangle(Ts)]
    for im,t in zip(ims,range(1,t+1)):
        Tsr=[dense.ising.ising_T(t,J,g,h,i) for J,g,h,i in zip(Js[:t],gs[:t],hs[:t],inits[:t])]
        assert im ==pytest.approx(list(dense.ising.im_rectangle(Tsr))[-1])
