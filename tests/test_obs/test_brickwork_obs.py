import pytest
import numpy as np
import imcode.dense as dense
import imcode.mps.brickwork as bw
import imcode.mps as mps
import scipy.linalg as la
from tenpy.networks.mpo import MPOEnvironment,MPO
import numpy.linalg as nla
from ..utils import seed_rng
MAX_T=3
SZ=np.array([[1.0,0.0],[0.0,-1.0]])
SX=np.array([[0.0,1.0],[1.0,0.0]])
SZ2=np.kron(SZ,np.eye(2))
SZ3=np.kron(SZ,np.eye(4))
SZ4=np.kron(SZ,np.eye(8))

def test_dense_brickwork_L2_obc():
    seed_rng("bw_L2_obc")
    gop=np.random.random((4,4))+np.random.random((4,4))*1.0j
    gop=gop+gop.T.conj()
    gop=la.expm(1.0j*gop)
    U=gop
    init=np.random.random((4,4))+np.random.random((4,4))*1.0j
    final=np.random.random((4,4))+np.random.random((4,4))*1.0j
    assert U==pytest.approx(dense.brickwork_F([U]))
    for t in range(1,MAX_T):
        S=dense.brickwork_Sb(t,U)
        B=dense.brickwork_La(t)
        assert B@S@B==pytest.approx(4.0)
        S=dense.brickwork_Sb(t,U,init=SZ2,final=SZ2)
        czzc=pytest.approx(np.trace(SZ2@nla.matrix_power(U,t)@SZ2@nla.matrix_power(U.T.conj(),t)))
        czz=B@S@B
        print(czz)
        assert czz==czzc
        S=dense.brickwork_Sb(t,U,init=init,final=final)
        czz=B@S@B
        print(czz)
        assert czz==pytest.approx(np.trace(init@nla.matrix_power(U,t)@final@nla.matrix_power(U.T.conj(),t)))
def test_dense_brickwork_L2_bbc():
    seed_rng("bw_L2_bbc")
    gop=np.random.random((4,4))+np.random.random((4,4))*1.0j
    gop=gop+gop.T.conj()
    gop=la.expm(1.0j*gop)
    lop1=np.random.random((2,2))+np.random.random((2,2))*1.0j
    lop1=lop1+lop1.T.conj()
    lop1=la.expm(1.0j*lop1)
    lop2=np.random.random((2,2))+np.random.random((2,2))*1.0j
    lop2=lop2+lop2.T.conj()
    lop2=la.expm(1.0j*lop2)
    init1=np.random.random((2,2))+np.random.random((2,2))*1.0j
    init2=np.random.random((2,2))+np.random.random((2,2))*1.0j
    final1=np.random.random((2,2))+np.random.random((2,2))*1.0j
    final2=np.random.random((2,2))+np.random.random((2,2))*1.0j
    # lop1=np.eye(2)
    # lop2=np.eye(2)
    U=np.kron(lop1,lop2)@gop
    assert U==pytest.approx(dense.brickwork_F([U]))
    for t in range(1,3):
        S=dense.brickwork_Sa(t,gop)
        B1=dense.brickwork_Lb(t,lop1)
        B2=dense.brickwork_Lb(t,lop2)
        assert B1@S@B2==pytest.approx(4.0)

        B1=dense.brickwork_Lb(t,lop1,init=SZ,final=SZ)
        B2=dense.brickwork_Lb(t,lop2)
        czz=B1@S@B2
        czzc=pytest.approx(np.trace(SZ2@nla.matrix_power(U,t)@SZ2@nla.matrix_power(U.T.conj(),t)))
        print((czz,czzc))
        assert czz==czzc

        B1=dense.brickwork_Lb(t,lop1,init=init1,final=final1)
        B2=dense.brickwork_Lb(t,lop2,init=init2,final=final2)
        czz=B1@S@B2
        czzc=np.trace(np.kron(init1,init2)@nla.matrix_power(U,t)@np.kron(final1,final2)@nla.matrix_power(U.T.conj(),t))
        assert czz==pytest.approx(czzc)

def test_mps_brickwork_L2_bbc():
    seed_rng("bw_L2_bbc")
    gop=np.random.random((4,4))+np.random.random((4,4))*1.0j
    gop=gop+gop.T.conj()
    gop=la.expm(1.0j*gop)
    lop1=np.random.random((2,2))+np.random.random((2,2))*1.0j
    lop1=lop1+lop1.T.conj()
    lop1=la.expm(1.0j*lop1)
    lop2=np.random.random((2,2))+np.random.random((2,2))*1.0j
    lop2=lop2+lop2.T.conj()
    lop2=la.expm(1.0j*lop2)
    init1=np.random.random((2,2))+np.random.random((2,2))*1.0j
    init2=np.random.random((2,2))+np.random.random((2,2))*1.0j
    final1=np.random.random((2,2))+np.random.random((2,2))*1.0j
    final2=np.random.random((2,2))+np.random.random((2,2))*1.0j
    U=np.kron(lop1,lop2)@gop
    # assert U==pytest.approx(dense.brickwork_F([U]))
    for t in range(1,10):
        S=bw.brickwork_Sa(t,gop)
        B1=bw.brickwork_Lb(t,lop1)
        B2=bw.brickwork_Lb(t,lop2)
        assert mps.embedded_obs(B1,S,B2) == pytest.approx(4.0)

        B1=bw.brickwork_Lb(t,lop1,init=SZ,final=SZ)
        B2=bw.brickwork_Lb(t,lop2)
        czz=mps.embedded_obs(B1,S,B2)
        czzc=pytest.approx(np.trace(SZ2@nla.matrix_power(U,t)@SZ2@nla.matrix_power(U.T.conj(),t)))
        print((czz,czzc))
        assert czz==czzc

        B1=bw.brickwork_Lb(t,lop1,init=init1,final=final1)
        B2=bw.brickwork_Lb(t,lop2,init=init2,final=final2)
        czz=mps.embedded_obs(B1,S,B2)
        czzc=np.trace(np.kron(init1,init2)@nla.matrix_power(U,t)@np.kron(final1,final2)@nla.matrix_power(U.T.conj(),t))
        assert czz==pytest.approx(czzc)
def test_mps_brickwork_L2_obc():
    seed_rng("bw_mps_L2_obc")
    gop=np.random.random((4,4))+np.random.random((4,4))*1.0j
    gop=gop+gop.T.conj()
    gop=la.expm(1.0j*gop)
    # gop=np.eye(4)
    U=gop
    init=np.random.random((4,4))+np.random.random((4,4))*1.0j
    final=np.random.random((4,4))+np.random.random((4,4))*1.0j
    # assert U==pytest.approx(dense.brickwork_F([U]))
    for t in range(1,10):
        S=bw.brickwork_Sb(t,U)
        B=bw.brickwork_La(t)
        assert mps.embedded_obs(B,S,B) == pytest.approx(4.0)
        S=bw.brickwork_Sb(t,U,init=SZ2,final=SZ2)
        czzc=pytest.approx(np.trace(SZ2@nla.matrix_power(U,t)@SZ2@nla.matrix_power(U.T.conj(),t)))
        czz=mps.embedded_obs(B,S,B)
        print((t,czz,czzc))
        assert czz==czzc
        S=bw.brickwork_Sb(t,U,init=init,final=final)
        czz=mps.embedded_obs(B,S,B)
        assert czz==pytest.approx(np.trace(init@nla.matrix_power(U,t)@final@nla.matrix_power(U.T.conj(),t)))
def test_dense_brickwork_L3():
    seed_rng("bw_L3")
    gop1=np.random.random((4,4))+np.random.random((4,4))*1.0j
    gop2=np.random.random((4,4))+np.random.random((4,4))*1.0j
    gop1=gop1+gop1.T.conj()
    gop1=la.expm(1.0j*gop1)
    gop2=gop2+gop2.T.conj()
    gop2=la.expm(1.0j*gop2)
    lop=np.random.random((2,2))+np.random.random((2,2))*1.0j
    lop=lop+lop.T.conj()
    lop=la.expm(1.0j*lop)

    init1=np.random.random((4,4))+np.random.random((4,4))*1.0j
    final1=np.random.random((4,4))+np.random.random((4,4))*1.0j
    init2=np.random.random((2,2))+np.random.random((2,2))*1.0j
    final2=np.random.random((2,2))+np.random.random((2,2))*1.0j

    U=dense.brickwork_F([gop1,np.kron(np.eye(2),lop)@gop2])
    for t in range(1,MAX_T):
        Sb=dense.brickwork_Sb(t,gop1)
        Sa=dense.brickwork_Sa(t,gop2)
        B1=dense.brickwork_La(t)
        B2=dense.brickwork_Lb(t,lop)
        assert B1@Sb@Sa@B2==pytest.approx(8)

        czzc=pytest.approx(np.trace(SZ3@nla.matrix_power(U,t)@SZ3@nla.matrix_power(U.T.conj(),t)))
        Sb=dense.brickwork_Sb(t,gop1,init=SZ2,final=SZ2)
        czz=B1@Sb@Sa@B2
        assert czz==czzc

        Sb=dense.brickwork_Sb(t,gop1,init=init1,final=final1)
        B2=dense.brickwork_Lb(t,lop,init=init2,final=final2)
        init=np.einsum("abcd,ef->abecdf",init1.reshape((2,2,2,2)),init2).reshape((8,8))
        final=np.einsum("abcd,ef->abecdf",final1.reshape((2,2,2,2)),final2).reshape((8,8))
        czz=B1@Sb@Sa@B2
        print(czz)
        assert czz==pytest.approx(np.trace(init@nla.matrix_power(U,t)@final@nla.matrix_power(U.T.conj(),t)))

def test_mps_brickwork_L3():
    return
    seed_rng("bw_mps_L3")
    gop1=np.random.random((4,4))+np.random.random((4,4))*1.0j
    gop2=np.random.random((4,4))+np.random.random((4,4))*1.0j
    gop1=gop1+gop1.T.conj()
    gop1=la.expm(1.0j*gop1)
    gop2=gop2+gop2.T.conj()
    gop2=la.expm(1.0j*gop2)
    lop=np.random.random((2,2))+np.random.random((2,2))*1.0j
    lop=lop+lop.T.conj()
    lop=la.expm(1.0j*lop)

    init1=np.random.random((4,4))+np.random.random((4,4))*1.0j
    final1=np.random.random((4,4))+np.random.random((4,4))*1.0j
    init2=np.random.random((2,2))+np.random.random((2,2))*1.0j
    final2=np.random.random((2,2))+np.random.random((2,2))*1.0j

    U=dense.brickwork_F([gop1,np.kron(np.eye(2),lop)@gop2])
    for t in range(1,10):
        Sb=bw.brickwork_Sb(t,gop1)
        Sa=bw.brickwork_Sa(t,gop2)
        B1=bw.brickwork_La(t)
        B2=bw.brickwork_Lb(t,lop)
        T=bw.brickwork_T(t,gop1,gop2)
        assert mps.embedded_obs(B1,T,B2)==pytest.approx(8)
        Ba=bw.brickwork_Lb(t,lop)
        mps.apply(T,Ba)
        assert mps.boundary_obs(B1,Ba)==pytest.approx(8)
        Ba=bw.brickwork_Lb(t,lop)
        mps.apply(Sa,Ba)
        mps.apply(Sb,Ba)
        assert mps.boundary_obs(B1,Ba)==pytest.approx(8)
        Ba=bw.brickwork_La(t)
        mps.apply(Sb,Ba)
        mps.apply(Sa,Ba)
        assert mps.boundary_obs(Ba,B2)==pytest.approx(8)

        czzc=pytest.approx(np.trace(SZ3@nla.matrix_power(U,t)@SZ3@nla.matrix_power(U.T.conj(),t)))
        Sb=bw.brickwork_Sb(t,gop1,init=SZ2,final=SZ2)
        T=bw.brickwork_T(t,gop1,gop2,init=SZ2,final=SZ2)
        assert mps.embedded_obs(B1,T,B2)==czzc
        Ba=bw.brickwork_Lb(t,lop)
        mps.apply(T,Ba)
        assert mps.boundary_obs(B1,Ba)==czzc
        Ba=bw.brickwork_Lb(t,lop)
        mps.apply(Sa,Ba)
        mps.apply(Sb,Ba)
        assert mps.boundary_obs(B1,Ba)==czzc
        Ba=bw.brickwork_La(t)
        mps.apply(Sb,Ba)
        mps.apply(Sa,Ba)
        assert mps.boundary_obs(Ba,B2)==czzc

        Sb=bw.brickwork_Sb(t,gop1,init=init1,final=final1)
        B2=bw.brickwork_Lb(t,lop,init=init2,final=final2)

        init=np.einsum("abcd,ef->abecdf",init1.reshape((2,2,2,2)),init2).reshape((8,8))
        final=np.einsum("abcd,ef->abecdf",final1.reshape((2,2,2,2)),final2).reshape((8,8))
        czzc=pytest.approx(np.trace(init@nla.matrix_power(U,t)@final@nla.matrix_power(U.T.conj(),t)))

        assert mps.embedded_obs(B1,T,B2)==czzc
        Ba=bw.brickwork_Lb(t,lop,init=init2,final=final2)
        mps.apply(T,Ba)
        assert mps.boundary_obs(B1,Ba)==czzc
        Ba=bw.brickwork_Lb(t,lop,init=init2,final=final2)
        mps.apply(Sa,Ba)
        mps.apply(Sb,Ba)
        assert mps.boundary_obs(B1,Ba)==czzc
        Ba=bw.brickwork_La(t)
        mps.apply(Sb,Ba)
        mps.apply(Sa,Ba)
        assert mps.boundary_obs(Ba,B2)==czzc


def test_dense_brickwork_L4_obc():
    return
    seed_rng("bw_L4_obc")
    gop1=np.random.random((4,4))+np.random.random((4,4))*1.0j
    gop2=np.random.random((4,4))+np.random.random((4,4))*1.0j
    gop3=np.random.random((4,4))+np.random.random((4,4))*1.0j
    gop1=gop1+gop1.T.conj()
    gop1=la.expm(1.0j*gop1)
    gop2=gop2+gop2.T.conj()
    gop2=la.expm(1.0j*gop2)
    gop3=gop3+gop3.T.conj()
    gop3=la.expm(1.0j*gop3)
    init1=np.random.random((4,4))+np.random.random((4,4))*1.0j
    final1=np.random.random((4,4))+np.random.random((4,4))*1.0j
    init2=np.random.random((4,4))+np.random.random((4,4))*1.0j
    final2=np.random.random((4,4))+np.random.random((4,4))*1.0j
    U=dense.brickwork_F([gop1,gop2,gop3])
    for t in range(1,MAX_T):
        Sb1=dense.brickwork_Sb(t,gop1)
        Sa=dense.brickwork_Sa(t,gop2)
        Sb2=dense.brickwork_Sb(t,gop3)
        B=dense.brickwork_La(t)
        assert B@Sb1@Sa@Sb2@B==pytest.approx(16)
        czzc=pytest.approx(np.trace(SZ4@nla.matrix_power(U,t)@SZ4@nla.matrix_power(U.T.conj(),t)))
        Sb1=dense.brickwork_Sb(t,gop1,init=SZ2,final=SZ2)
        czz=B@Sb1@Sa@Sb2@B
        print(czz)
        assert czz==czzc

        Sb1=dense.brickwork_Sb(t,gop1,init=init1,final=final1)
        Sb2=dense.brickwork_Sb(t,gop3,init=init2,final=final2)
        init=np.kron(init1,init2)
        final=np.kron(final1,final2)
        czz=B@Sb1@Sa@Sb2@B
        assert czz==pytest.approx(np.trace(init@nla.matrix_power(U,t)@final@nla.matrix_power(U.T.conj(),t)))
