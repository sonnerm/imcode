import imcode.dense as dense
import imcode.sparse as sparse
import imcode.mps as mps
import pytest
from ..utils import seed_rng
import numpy as np

@pytest.fixture
def dense_heisenberg_czz():
    t=3
    seed_rng("dense_heisenberg_czz")
    Jx,Jy,Jz,hx,hy,hz=np.random.normal(size=6)
    Jx=Jy=Jz=hx=hy=hz=0
    i=np.random.random()
    T=dense.heisenberg_T(t,Jx,Jy,Jz,hx,hy,hz,(i,1.0-i))
    im=dense.im_finite([T]*(2*t),dense.brickwork_open_boundary_im(t))
    lop=dense.heisenberg_L(t,hx,hy,hz,(i,1.0-i))@dense.brickwork_zz_operator(t)
    return (dense.embedded_obs(im,lop,im),dense.embedded_obs(im,lop,dense.brickwork_open_boundary_im(t)),(t,Jx,Jy,Jz,hx,hy,hz,i))



def test_dense_direct_heisenberg(dense_heisenberg_czz):
    t,Jx,Jy,Jz,hx,hy,hz,i=dense_heisenberg_czz[-1]
    L=7
    F=dense.heisenberg_F([Jx]*(L-1),[Jy]*(L-1),[Jz]*(L-1),[hx]*L,[hy]*L,[hz]*L)
    init=dense.dense_kron([np.array([[1.0-i,0.0],[0.0,i]])]*7)
    assert dense.direct_czz(F,t,3,3,init=init)==pytest.approx(dense_heisenberg_czz[0])
    assert dense.direct_czz(F,t,0,0,init=init)==pytest.approx(dense_heisenberg_czz[1])
# @pytest.mark.xfail
# def test_sparse_direct_heisenberg_czz(dense_heisenberg_czz):
#     t,J,g,h=dense_heisenberg_czz[-1]
#     L=7
#     F=sparse.heisenberg_F([J]*(L-1),[g]*L,[h]*L)
#     assert sparse.direct_czz(F,t,3,3)==pytest.approx(dense_heisenberg_czz[0])
#     assert sparse.direct_czz(F,t,0,0)==pytest.approx(dense_heisenberg_czz[1])

# def test_mps_direct_heisenberg_czz_pol(dense_heisenberg_czz_pol):
#     t,J,g,h=dense_heisenberg_czz_pol[-1]
#     L=7
#     F=mps.heisenberg_F([J]*(L-1),[g]*L,[h]*L)
#
#     # evop=mps.evolve_operator(zz_op,F)
#     assert mps.direct_czz(F,t,3,3)[0]==pytest.approx(dense_heisenberg_czz_pol[0])
#     assert mps.direct_czz(F,t,0,0)[0]==pytest.approx(dense_heisenberg_czz_pol[1])

# def test_sparse_heisenberg_czz_pol(dense_heisenberg_czz_pol):
#     t,J,g,h=dense_heisenberg_czz_pol[-1]
#     T=sparse.heisenberg_T(t,J,g,h,(1.0,0.0))
#     im=sparse.im_iterative(T)
#     lop=sparse.heisenberg_W(t,g,(1.0,0.0))@sparse.heisenberg_h(t,h)
#     assert sparse.embedded_czz(im,lop) == pytest.approx(dense_heisenberg_czz_pol[0])
#     assert sparse.boundary_czz(im,lop) == pytest.approx(dense_heisenberg_czz_pol[1])
#
# def test_mps_heisenberg_czz_pol(dense_heisenberg_czz_pol):
#     t,J,g,h=dense_heisenberg_czz_pol[-1]
#     T=mps.fold.heisenberg_T(t,J,g,h,(1.0,0.0))
#     im=mps.im_iterative(T)
#     lop=mps.multiply_mpos([mps.fold.heisenberg_W(t,g,(1.0,0.0)),mps.fold.heisenberg_h(t,h)])
#     # st=mps.zz_state(t)
#     # mps.apply(lop,st)
#     # assert mps.mpo_to_dense(lop)@dense.zz_state(t) == pytest.approx(mps.mps_to_dense(st))
#     # dst=st.get_theta(0, st.L).take_slice([0, 0], ['vL', 'vR']).to_ndarray().ravel()*st.norm
#     # dim=im.get_theta(0, im.L).take_slice([0, 0], ['vL', 'vR']).to_ndarray().ravel()*im.norm
#     # assert st.overlap(im) == pytest.approx(dst.conj()@dim)
#     # assert st.overlap(im) == pytest.approx(mps.mps_to_dense(st).conj()@mps.mps_to_dense(im))
#     # assert mps.mps_to_dense(mps.zz_state(t)) == pytest.approx(dense.zz_state(t))
#     assert mps.boundary_czz(im,lop) == pytest.approx(dense_heisenberg_czz_pol[1])
#     assert mps.embedded_czz(im,lop) == pytest.approx(dense_heisenberg_czz_pol[0])
