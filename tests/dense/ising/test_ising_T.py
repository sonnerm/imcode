import numpy as np
import numpy.linalg as la
from imcode import dense
import pytest
def test_dense_ising_h_real(seed_rng):
    T=3
    h=np.random.normal()
    dih=dense.ising.ising_h(T,h)
    assert dih.dtype==np.complex_
    assert dih.shape==(64,64)
    assert np.diag(np.diag(dih))==pytest.approx(dih) #diagonal
    assert dih.conj()*dih==pytest.approx(np.eye(dih.shape[0])) #unitary
    assert dih.conj()!=pytest.approx(dih)
def test_dense_ising_h_zero():
    T=3
    h=0.0
    dih=dense.ising.ising_h(T,h)
    assert dih==pytest.approx(np.eye(64))

def test_dense_ising_h_complex(seed_rng):
    T=3
    h=np.random.normal()+np.random.normal()*1.0j
    dih=dense.ising.ising_h(T,h)
    assert dih.dtype==np.complex_
    assert dih.shape==(64,64)
    assert np.diag(np.diag(dih))==pytest.approx(dih) #diagonal
    assert dih.conj()*dih!=pytest.approx(np.eye(dih.shape[0]))
    assert dih.conj()!=pytest.approx(dih)

def test_dense_ising_W_real(seed_rng):
    T=3
    g=np.random.normal()
    init=np.random.normal(size=(2,2))+1.0j*np.random.normal(size=(2,2))
    final=np.random.normal(size=(2,2))+1.0j*np.random.normal(size=(2,2))
    init=init.T.conj()+init #init hermitian
    final=final.T.conj()+final #final hermitian
    diW=dense.ising.ising_W(T,g,init,final)
    assert diW.dtype==np.complex_
    assert diW.shape==(64,64)
    assert np.diag(np.diag(diW))==pytest.approx(diW) #diagonal
    assert diW.conj()*diW!=pytest.approx(np.eye(64)) #generic
    assert diW.conj()!=pytest.approx(diW)

def test_dense_ising_W_complex(seed_rng):
    T=3
    g=np.random.normal()+1.0j*np.random.normal()
    init=np.random.normal(size=(2,2))+1.0j*np.random.normal(size=(2,2))
    final=np.random.normal(size=(2,2))+1.0j*np.random.normal(size=(2,2))
    diW=dense.ising.ising_W(T,g,init,final)
    assert diW.dtype==np.complex_
    assert diW.shape==(64,64)
    assert np.diag(np.diag(diW))==pytest.approx(diW) #diagonal
    assert diW.conj()*diW!=pytest.approx(np.eye(diW.shape[0]))
    assert diW.conj()!=pytest.approx(diW)
def test_dense_ising_W_T1(seed_rng):
    T=1
    g=np.random.normal()
    diW=dense.ising.ising_W(T,g)

def test_dense_ising_W_zero():
    T=3
    g=0.0
    diW=dense.ising.ising_W(T,g)
    diWc=np.zeros((4**T,4**T))
    diWc[0,0]=0.5
    diWc[-1,-1]=0.5
    assert diW==pytest.approx(diWc)

@pytest.mark.skip
def test_dense_ising_W_pi4():
    #dual unitary case, not 100% sure what to expect
    T=2
    g=np.pi/4
    diW=dense.ising.ising_W(T,g,init=np.ones((2,2)),final=np.ones((2,2)))*4
    print(np.diag(diW))
    print(np.diag(diW.conj())*np.diag(diW))
    assert diW.dtype==np.complex_
    assert diW.shape==(4**T,4**T)
    assert np.diag(np.diag(diW))==pytest.approx(diW) #diagonal
    assert diW.conj()*diW==pytest.approx(np.eye(diW.shape[0])) #unitary
    assert diW.conj()!=pytest.approx(diW)

def test_dense_ising_J_complex(seed_rng):
    T=3
    J=np.random.normal()+1.0j*np.random.normal()
    diJ=dense.ising.ising_J(T,J)
    assert diJ.dtype==np.complex_
    assert diJ.shape==(64,64)
    assert diJ.conj()*diJ!=pytest.approx(np.eye(diJ.shape[0]))
    assert diJ.conj()!=pytest.approx(diJ)
    #TODO: test separable
def test_dense_ising_J_real(seed_rng):
    T=3
    J=np.random.normal()
    diJ=dense.ising.ising_J(T,J)
    assert diJ.dtype==np.complex_
    assert diJ.shape==(64,64)
    assert diJ.conj()*diJ!=pytest.approx(np.eye(diJ.shape[0]))
    assert diJ.conj()!=pytest.approx(diJ)
def test_dense_ising_J_zero(seed_rng):
    T=3
    J=0.0
    diJ=dense.ising.ising_J(T,J)
    assert diJ==pytest.approx(np.ones((64,64)))

def test_dense_ising_T_real(seed_rng):
    T=1
    J=np.random.normal()+1.0j*np.random.normal()
    g=np.random.normal()+1.0j*np.random.normal()
    h=np.random.normal()+1.0j*np.random.normal()
    init=np.random.normal(size=(2,2))+1.0j*np.random.normal(size=(2,2))
    init=init.T.conj()+init
    init=init@init
    init/=np.trace(init)#init is a proper density matrix
    print(init)
    diT=dense.ising.ising_T(T,J,g,h,init)
    print(diT)
    assert diT.dtype==np.complex_
    assert diT.shape==(4**T,4**T)
    print(la.eigvals(diT))
    assert np.sort(la.eigvals(diT).real)==pytest.approx(np.sort([0.0]*(4**T-1)+[1.0])) #IM gets checked elsewhere
    assert la.eigvals(diT).imag==pytest.approx(np.zeros((4**T))) #IM gets checked elsewhere

def test_dense_ising_T_pi4():
    T=3
    J=np.pi/4
    g=np.pi/4
    h=1.0
    init=np.random.normal(size=(2,2))+1.0j*np.random.normal(size=(2,2))
    init=init.T.conj()+init
    init=init@init
    init/=np.trace(init)#init is a proper density matrix
    diT=dense.ising.ising_T(T,J,g,h,init)
    assert diT.dtype==np.complex_
    assert diT.shape==(64,64)
    assert np.sort(la.eigvals(diT))==pytest.approx(np.sort([0.0]*63+[1.0])) #IM gets checked elsewhere
