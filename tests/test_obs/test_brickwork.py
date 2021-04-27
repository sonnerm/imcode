import pytest
import numpy as np
import imcode.dense as dense
import imcode.mps.brickwork as bw
import scipy.linalg as la
import numpy.linalg as nla
from ..utils import seed_rng

SZ=np.array([[1.0,0.0],[0.0,-1.0]])
SZ2=np.kron(SZ,np.eye(2))/2
SZ3=np.kron(SZ,np.eye(4))/np.sqrt(8)

def test_dense_brickwork_L1():
    seed_rng("bw_L1")
    lop=np.random.random((2,2))+np.random.random((2,2))*1.0j
    lop=la.expm(1.0j*(lop+lop.T.conj()))
    # assert dense.brickwork_F([np.eye(4)],[lop]) == pytest.approx(lop)
    for t in range(1,2):
        L=dense.brickwork_L(t,lop)
        B=dense.brickwork_open_boundary_im(t)
        assert B@L@B==pytest.approx(1.0)
        Lz=dense.brickwork_L(t,lop,init=(0.5,0.0,0.0,-0.5),final=(1.0,0.0,0.0,-1.0))
        assert B@Lz@B==pytest.approx(np.trace(SZ@nla.matrix_power(lop,t)@(SZ/2)@nla.matrix_power(lop.T.conj(),t)))
def test_mps_brickwork_L1():
    seed_rng("bw_L1")
    lop=np.random.random((2,2))+np.random.random((2,2))*1.0j
    lop=la.expm(1.0j*(lop+lop.T.conj()))
    for t in range(1,2):
        L=bw.brickwork_L(t,lop)
        B=bw.brickwork_open_boundary_im(t)
        assert B@L@B==pytest.approx(1.0)
        Lz=dense.brickwork_L(t,lop,init=(0.5,0.0,0.0,-0.5),final=(1.0,0.0,0.0,-1.0))
        assert B@Lz@B==pytest.approx(np.trace(SZ@nla.matrix_power(lop,t)@(SZ/2)@nla.matrix_power(lop.T.conj(),t)))

def test_dense_brickwork_L2():
    seed_rng("bw_L2")
    lop1=np.random.random((2,2))+np.random.random((2,2))*1.0j
    lop2=np.random.random((2,2))+np.random.random((2,2))*1.0j
    gop=np.random.random((4,4))+np.random.random((4,4))*1.0j
    lop1=la.expm(1.0j*(lop1+lop1.T.conj()))
    lop2=la.expm(1.0j*(lop2+lop2.T.conj()))
    gop=la.expm(1.0j*(gop+gop.T.conj()))
    # gop=np.eye(4)
    U=np.kron(lop1,lop2)@gop
    assert U==pytest.approx(dense.brickwork_F([gop,np.eye(4)],[lop1,lop2]))
    for t in range(1,3):
        L1=dense.brickwork_L(t,lop1)
        L2=dense.brickwork_L(t,lop2)
        S=dense.brickwork_S(t,gop)
        B=dense.brickwork_open_boundary_im(t)
        assert B@L1@S@L2@B==pytest.approx(1.0)
        Lz1=dense.brickwork_L(t,lop1,init=(0.5,0.0,0.0,-0.5),final=(1.0,0.0,0.0,-1.0))
        czz=B@Lz1@S@L2@B
        print(czz)
        assert czz==pytest.approx(np.trace(SZ2@nla.matrix_power(U,t)@SZ2@nla.matrix_power(U.T.conj(),t)))

def test_mps_brickwork_L2():
    seed_rng("bw_L2")
    lop1=np.random.random((2,2))+np.random.random((2,2))*1.0j
    lop2=np.random.random((2,2))+np.random.random((2,2))*1.0j
    gop=np.random.random((4,4))+np.random.random((4,4))*1.0j
    lop1=la.expm(1.0j*(lop1+lop1.T.conj()))
    lop2=la.expm(1.0j*(lop2+lop2.T.conj()))
    gop=la.expm(1.0j*(gop+gop.T.conj()))
    # gop=np.eye(4)
    U=np.kron(lop1,lop2)@gop
    for t in range(1,3):
        L1=mps.brickwork_L(t,lop1)
        L2=mps.brickwork_L(t,lop2)
        S=mps.brickwork_S(t,gop)
        B=mps.brickwork_open_boundary_im(t)
        assert B@L1@S@L2@B==pytest.approx(1.0)
        Lz1=bw.brickwork_L(t,lop1,init=(0.5,0.0,0.0,-0.5),final=(1.0,0.0,0.0,-1.0))
        czz=B@Lz1@S@L2@B
        print(czz)
        assert czz==pytest.approx(np.trace(SZ2@nla.matrix_power(U,t)@SZ2@nla.matrix_power(U.T.conj(),t)))

def test_brickwork_L3():
    seed_rng("bw_L3")
    lop1=np.random.random((2,2))+np.random.random((2,2))*1.0j
    lop2=np.random.random((2,2))+np.random.random((2,2))*1.0j
    lop3=np.random.random((2,2))+np.random.random((2,2))*1.0j
    gop1=np.random.random((4,4))+np.random.random((4,4))*1.0j
    gop2=np.random.random((4,4))+np.random.random((4,4))*1.0j
    lop1=la.expm(1.0j*(lop1+lop1.T.conj()))
    lop2=la.expm(1.0j*(lop2+lop2.T.conj()))
    # lop1=np.eye(2)
    lop2=np.eye(2)
    lop3=np.eye(2)
    lop3=la.expm(1.0j*(lop3+lop3.T.conj()))
    gop1=la.expm(1.0j*(gop1+gop1.T.conj()))
    gop2=la.expm(1.0j*(gop2+gop2.T.conj()))
    U=np.kron(gop1,np.eye(2))@np.kron(np.eye(2),gop2)@np.kron(lop1,np.kron(lop2,lop3))
    # assert U==pytest.approx(dense.brickwork_F([gop1,gop2,np.eye(4)],[lop1,lop2,lop3]))
    for t in range(1,4):
        L1=dense.brickwork_L(t,lop1)
        L2=dense.brickwork_L(t,lop2)
        L3=dense.brickwork_L(t,lop3)
        S1=dense.brickwork_S(t,gop1)
        S2=dense.brickwork_S(t,gop2)
        B=dense.brickwork_open_boundary_im(t)
        assert B@L1@S1@L2@S2@L3@B==pytest.approx(1.0)
        Lz1=dense.brickwork_L(t,lop1,init=(0.5,0.0,0.0,-0.5),final=(1.0,0.0,0.0,-1.0))
        czz=B@Lz1@S1@L2@S2@L3@B
        print(czz)
        assert czz==pytest.approx(np.trace(SZ3@nla.matrix_power(U,t)@SZ3@nla.matrix_power(U.T.conj(),t)))
