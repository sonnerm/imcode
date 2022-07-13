import imcode
import pytest
import ttarray as tt
import numpy as np
from imcode import SZ,SX,ID,ZE
from .. import check_model
import functools
def mkron(a):
    return functools.reduce(np.kron,a)

def mouter(a):
    return functools.reduce(np.outer,a).ravel()
def test_product_homhom(seed_rng):
    L=7
    t=6
    J=np.random.random()-0.5
    g=np.random.random()-0.5
    h=np.random.random()-0.5
    init=[np.random.random((2,))+np.random.random((2,))*1.0j-0.5-0.5j for _ in range(L)]
    init=[i.T.conj()+i for i in init]
    init=[i/np.sqrt(np.sum(i.conj()*i)) for i in init]
    init=[init[0] for _ in init]
    init=tt.fromproduct([np.outer(i.T.conj(),i) for i in init])
    F=imcode.ising_F(L,J,g,h)
    Fs=[F for _ in range(t)]
    T=imcode.ising_T(t,J,g,h)
    Ts=[T for _ in range(t)]
    ch1=np.array(imcode.unitary_channel(imcode.ising_F(1,J,g,h)))
    ch2=np.array(imcode.unitary_channel(imcode.ising_F(2,J,g,h)))
    check_model(L,t,init,Fs,Ts,Ts,imcode.zoz_lcga,ch1,ch2,ch1,ch2,imcode.ising_boundary_evolution,imcode.ising_embedded_evolution)

def test_mps_hethom():
    pass

def test_mps_hethet():
    pass