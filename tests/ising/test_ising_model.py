import imcode
import pytest
import ttarray as tt
import numpy as np
from imcode import SZ,SX,ID,ZE
from .. import check_model
@pytest.mark.skip
def test_product_homhom(seed_rng):
    L=7
    t=6
    J=np.random.random()-0.5
    g=np.random.random()-0.5
    h=np.random.random()-0.5
    init=[np.random.random((2,))+np.random.random((2,))*1.0j-0.5-0.5j for _ in range(L)]
    init=[i.T.conj()+i for i in init]
    init=[i/np.sqrt(np.sum(i.conj()*i)) for i in init]
    init=tt.fromproduct([np.outer(i.T.conj(),i) for i in init])
    F=imcode.ising_F(L,J,g,h)
    Fs=[F for _ in range(t)]
    T=imcode.ising_T(t,J,g,h)
    Ts=[T for _ in range(t)]
    ch1=np.array(imcode.unitary_channel(imcode.ising_F(1,J,g,h)))
    ch2=np.array(imcode.unitary_channel(imcode.ising_F(2,J,g,h)))
    #rectangle
    check_model(L,t,init,Fs,Ts,Ts,imcode.zoz_lcga,ch1,ch2,ch1,ch2,imcode.ising_boundary_evolution,imcode.ising_embedded_evolution)
    #lcga
    Ts=[imcode.ising_T(t,J,g,h) for t in range(1,t+1)]+[T]*(L-t)
    check_model(L,t,init,Fs,Ts,Ts,imcode.zoz_lcga,ch1,ch2,ch1,ch2,imcode.ising_boundary_evolution,imcode.ising_embedded_evolution)
@pytest.mark.skip
def test_product_hethom(seed_rng):
    L=7
    t=6
    J=np.random.random(size=(L-1,))-0.5
    g=np.random.random(size=(L,))-0.5
    h=np.random.random(size=(L,))-0.5
    init=[np.random.random((2,))+np.random.random((2,))*1.0j-0.5-0.5j for _ in range(L)]
    init=[i.T.conj()+i for i in init]
    init=[i/np.sqrt(np.sum(i.conj()*i)) for i in init]
    init=tt.fromproduct([np.outer(i.T.conj(),i) for i in init])
    F=imcode.ising_F(L,J,g,h)
    Fs=[F for _ in range(t)]
    Tsl=[imcode.ising_T(t,J[i],g[i],h[i]) for i in range(L-1)]
    Tsr=[imcode.ising_T(t,J[i-1],g[i],h[i]) for i in range(L-1,0,-1)]
    chl=np.array(imcode.unitary_channel(imcode.ising_F(1,0,g[0],h[0])))
    chr2=np.array(imcode.unitary_channel(imcode.ising_F(2,J[-1],g[-2:],h[-2:])))
    che=np.array(imcode.unitary_channel(imcode.ising_F(1,0,g[L//2],h[L//2])))
    che2=np.array(imcode.unitary_channel(imcode.ising_F(2,J[L//2],g[L//2:L//2+2],h[L//2:L//2+2])))
    #rectangle
    check_model(L,t,init,Fs,Tsl,Tsr,imcode.zoz_lcga,chl,chr2,che,che2,imcode.ising_boundary_evolution,imcode.ising_embedded_evolution)
    #lcga
    Tsl=[imcode.ising_T(min(i+1,t),J[i],g[i],h[i]) for i in range(L-1)]
    Tsr=[imcode.ising_T(min(L-i,t),J[i-1],g[i],h[i]) for i in range(L-1,0,-1)]
    check_model(L,t,init,Fs,Tsl,Tsr,imcode.zoz_lcga,chl,chr2,che,che2,imcode.ising_boundary_evolution,imcode.ising_embedded_evolution)
@pytest.mark.skip
def test_product_hethet(seed_rng):
    L=7
    t=6
    J=np.random.random(size=(L-1,t))-0.5
    g=np.random.random(size=(L,t))-0.5
    h=np.random.random(size=(L,t))-0.5
    init=[np.random.random((2,))+np.random.random((2,))*1.0j-0.5-0.5j for _ in range(L)]
    init=[i.T.conj()+i for i in init]
    init=[i/np.sqrt(np.sum(i.conj()*i)) for i in init]
    init=tt.fromproduct([np.outer(i.T.conj(),i) for i in init])
    Fs=[imcode.ising_F(L,J,g,h) for J,g,h in zip(J.T,g.T,h.T)]
    Tsl=[imcode.ising_T(t,J[i],g[i],h[i]) for i in range(L-1)]
    Tsr=[imcode.ising_T(t,J[i-1],g[i],h[i]) for i in range(L-1,0,-1)]
    chl=[np.array(imcode.unitary_channel(imcode.ising_F(1,0,g[0,i],h[0,i]))) for i in range(t)]
    chr2=[np.array(imcode.unitary_channel(imcode.ising_F(2,J[-1,i],g[-2:,i],h[-2:,i]))) for i in range(t)]
    che=[np.array(imcode.unitary_channel(imcode.ising_F(1,0,g[L//2,i],h[L//2,i]))) for i in range(t)]
    che2=[np.array(imcode.unitary_channel(imcode.ising_F(2,J[L//2,i],g[L//2:L//2+2,i],h[L//2:L//2+2,i]))) for i in range(t)]
    #rectangle
    check_model(L,t,init,Fs,Tsl,Tsr,imcode.zoz_lcga,chl,chr2,che,che2,imcode.ising_boundary_evolution,imcode.ising_embedded_evolution)
    #lcga
    Tsl=[imcode.ising_T(min(i+1,t),J[i],g[i],h[i]) for i in range(L-1)]
    print([T.shape for T in Tsl])
    Tsr=[imcode.ising_T(min(L-i,t),J[i-1],g[i],h[i]) for i in range(L-1,0,-1)]
    check_model(L,t,init,Fs,Tsl,Tsr,imcode.zoz_lcga,chl,chr2,che,che2,imcode.ising_boundary_evolution,imcode.ising_embedded_evolution)
@pytest.mark.skip
def test_mps_homhom():

    L=7
    t=6
    J=np.random.random()-0.5
    g=np.random.random()-0.5
    h=np.random.random()-0.5
    p=np.random.random()
    init1=np.random.random((2**L,))+np.random.random((2**L,))*1.0j
    init1/=np.sqrt(np.sum(init1.conj()*init1))
    init1=tt.frommatrices([np.einsum("abc,def->adbecf",i,i.conj()).reshape((i.shape[0]**2,2,2,i.shape[-1]**2)) for i in tt.array(init1).tomatrices()])
    init=init1.canonicalize()
    # init1=init1.canonicalize()*p
    # init2=np.random.random((2**L,))+np.random.random((2**L,))*1.0j
    # init2/=np.sqrt(np.sum(init2.conj()*init2))
    # init2=init2.canonicalize()*(1-p)
    # init=init1+init2
    # init=init.canonicalize()
    #true mixed state

    F=imcode.ising_F(L,J,g,h)
    Fs=[F for _ in range(t)]
    T=imcode.ising_T(t,J,g,h)
    Ts=[T for _ in range(t)]
    ch1=np.array(imcode.unitary_channel(imcode.ising_F(1,J,g,h)))
    ch2=np.array(imcode.unitary_channel(imcode.ising_F(2,J,g,h)))
    #rectangle
    check_model(L,t,init,Fs,Ts,Ts,imcode.zoz_lcga,ch1,ch2,ch1,ch2,imcode.ising_boundary_evolution,imcode.ising_embedded_evolution)
    #lcga
    Ts=[imcode.ising_T(t,J,g,h) for t in range(1,t+1)]+[T]*(L-t)
    check_model(L,t,init,Fs,Ts,Ts,imcode.zoz_lcga,ch1,ch2,ch1,ch2,imcode.ising_boundary_evolution,imcode.ising_embedded_evolution)
#
@pytest.mark.skip
def test_mps_hethet():

    L=7
    t=6
    J=np.random.random(size=(L-1,t))-0.5
    g=np.random.random(size=(L,t))-0.5
    h=np.random.random(size=(L,t))-0.5
    p=np.random.random()
    init=[np.random.random(size=(16,2,2,16)) for _ in range(L)]
    init[0]=init[0][:1,...]
    init[-1]=init[-1][...,-1:]
    init=tt.frommatrices(init)
    init=init.canonicalize()
    Fs=[imcode.ising_F(L,J,g,h) for J,g,h in zip(J.T,g.T,h.T)]
    Tsl=[imcode.ising_T(t,J[i],g[i],h[i]) for i in range(L-1)]
    Tsr=[imcode.ising_T(t,J[i-1],g[i],h[i]) for i in range(L-1,0,-1)]
    chl=[np.array(imcode.unitary_channel(imcode.ising_F(1,0,g[0,i],h[0,i]))) for i in range(t)]
    chr2=[np.array(imcode.unitary_channel(imcode.ising_F(2,J[-1,i],g[-2:,i],h[-2:,i]))) for i in range(t)]
    che=[np.array(imcode.unitary_channel(imcode.ising_F(1,0,g[L//2,i],h[L//2,i]))) for i in range(t)]
    che2=[np.array(imcode.unitary_channel(imcode.ising_F(2,J[L//2,i],g[L//2:L//2+2,i],h[L//2:L//2+2,i]))) for i in range(t)]
    #rectangle
    check_model(L,t,init,Fs,Tsl,Tsr,imcode.zoz_lcga,chl,chr2,che,che2,imcode.ising_boundary_evolution,imcode.ising_embedded_evolution,np.array([1,1,1,1]))
    #lcga
    Tsl=[imcode.ising_T(min(i+1,t),J[i],g[i],h[i]) for i in range(L-1)]
    print([T.shape for T in Tsl])
    Tsr=[imcode.ising_T(min(L-i,t),J[i-1],g[i],h[i]) for i in range(L-1,0,-1)]
    check_model(L,t,init,Fs,Tsl,Tsr,imcode.zoz_lcga,chl,chr2,che,che2,imcode.ising_boundary_evolution,imcode.ising_embedded_evolution,np.array([1,1,1,1]))
