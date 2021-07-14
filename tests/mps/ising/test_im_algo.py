import imcode.mps as mps
import numpy as np
import imcode.dense as dense
import pytest
def test_im_rectangle_dense(seed_rng):
    t=5
    Js,gs,hs=np.random.normal(size=(3,t))
    inits=[np.random.normal(size=(2,2))+1.0j*np.random.normal(size=(2,2)) for _ in range(t)]
    Tsm=[mps.ising.ising_T(t,J,g,h,i) for J,g,h,i in zip(Js,gs,hs,inits)]
    Tsd=[dense.ising.ising_T(t,J,g,h,i) for J,g,h,i in zip(Js,gs,hs,inits)]
    imsm=[im for im in mps.ising.im_rectangle(Tsm)]
    imsd=[im for im in dense.ising.im_rectangle(Tsd)]
    for imm,imd in zip(imsm,imsd):
        assert imm.to_dense() == pytest.approx(imd)


def test_im_diamond_dense(seed_rng):
    J,g,h=np.random.normal(size=3)
    t=5
    Tsm=[mps.ising.ising_T(t,J,g,h) for t in range(1,6,2)]
    Tsd=[dense.ising.ising_T(t,J,g,h) for t in range(1,6,2)]
    imsm=[im for im in mps.ising.im_diamond(Tsm)]
    imsd=[im for im in dense.ising.im_diamond(Tsd)]
    for imm,imd in zip(imsm,imsd):
        assert imm.to_dense() == pytest.approx(imd)

def test_im_triangle_dense(seed_rng):
    t=5
    Js,gs,hs=np.random.normal(size=(3,t))
    inits=[np.random.normal(size=(2,2))+1.0j*np.random.normal(size=(2,2)) for i in range(t)]
    inits=[i.T.conj()@i for i in inits]
    inits=[i/np.trace(i) for i in inits]
    Tsm=[mps.ising.ising_T(t,J,g,h,i) for t,J,g,h,i in zip(range(1,t+1),Js,gs,hs,inits)]
    Tsd=[dense.ising.ising_T(t,J,g,h,i) for t,J,g,h,i in zip(range(1,t+1),Js,gs,hs,inits)]
    imsm=[im for im in mps.ising.im_triangle(Tsm)]
    imsd=[im for im in dense.ising.im_triangle(Tsd)]
    for imm,imd in zip(imsm,imsd):
        assert imm.to_dense()==pytest.approx(imd)

def test_im_rectangle(seed_rng):
    J,g,h=np.random.normal(size=3)
    t=10
    T=mps.ising.ising_T(t,J,g,h)
    ims=[im for im in mps.ising.im_rectangle(T,chi=64)]
    assert ims[-1].conj()@ims[-2]==pytest.approx(np.sqrt((ims[-1].conj()@ims[-1])*(ims[-2].conj()@ims[-2]))) #convergence achieved
def test_im_triangle_het(seed_rng):
    t=10
    Js,gs,hs=np.random.normal(size=(3,t))
    inits=[np.random.normal(size=(2,2))+1.0j*np.random.normal(size=(2,2)) for i in range(t)]
    inits=[i.T.conj()@i for i in inits]
    inits=[i/np.trace(i) for i in inits]
    Ts=[mps.ising.ising_T(t,J,g,h,i) for t,J,g,h,i in zip(range(1,t+1),Js,gs,hs,inits)]
    ims=[im for im in mps.ising.im_triangle(Ts,chi=64)]
    for im,t in zip(ims,range(1,t+1)):
        Tsr=[mps.ising.ising_T(t,J,g,h,i) for J,g,h,i in zip(Js[:t],gs[:t],hs[:t],inits[:t])]
        imo=list(mps.ising.im_rectangle(Tsr,chi=64))[-1]
        assert im.conj()@imo==pytest.approx(np.sqrt((im.conj()@im)*(imo.conj()@imo))) #convergence achieved
def test_im_diamond_het(seed_rng):
    t=10
    Js,gs,hs=np.random.normal(size=(3,t))
    Ts=[mps.ising.ising_T(t,J,g,h) for t,J,g,h in zip(range(1,t+1),Js,gs,hs)]
    ims=[im for im in mps.ising.im_triangle(Ts,chi=64)]
    for im,t in zip(ims,range(1,t+1)):
        Tsr=[mps.ising.ising_T(t,J,g,h) for J,g,h in zip(Js[:t],gs[:t],hs[:t])]
        imo=list(mps.ising.im_rectangle(Tsr,chi=64))[-1]
        assert im.conj()@imo==pytest.approx(np.sqrt((im.conj()@im)*(imo.conj()@imo))) #convergence achieved
