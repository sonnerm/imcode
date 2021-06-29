# import numpy as np
# from imcode import dense
# from imcode import sparse
# from imcode import mps
#
# import pytest
# # pytestmark=pytest.mark.skip("skip everything")
# @pytest.fixture(scope="module")
# def dense_ising_F_complex():
#     L=5
#     seed_rng("dense_ising_F_complex")
#     J=np.random.normal(size=(L,))+np.random.normal(size=(L,))*1.0j
#     g=np.random.normal(size=(L,))+np.random.normal(size=(L,))*1.0j
#     h=np.random.normal(size=(L,))+np.random.normal(size=(L,))*1.0j
#     return (dense.ising_F(J,g,h),(J,g,h))
# @pytest.fixture(scope="module")
# def dense_ising_F_complex_obc():
#     L=5
#     seed_rng("dense_ising_F_complex_obc")
#     J=np.random.normal(size=(L-1,))+np.random.normal(size=(L-1,))*1.0j
#     g=np.random.normal(size=(L,))+np.random.normal(size=(L,))*1.0j
#     h=np.random.normal(size=(L,))+np.random.normal(size=(L,))*1.0j
#     return (dense.ising_F(J,g,h),(J,g,h))
#
# @pytest.fixture(scope="module")
# def dense_ising_F_real():
#     L=5
#     seed_rng("dense_ising_F_real")
#     J=np.random.normal(size=(L,))
#     g=np.random.normal(size=(L,))
#     h=np.random.normal(size=(L,))
#     return (dense.ising_F(J,g,h),(J,g,h))
#
# def test_dense_ising_F_real(dense_ising_F_real):
#     diF=dense_ising_F_real[0]
#     assert diF.dtype==np.complex_
#     assert diF.conj().T@diF==pytest.approx(np.eye(diF.shape[0]))
#     assert diF@diF.T.conj()==pytest.approx(np.eye(diF.shape[0]))
#     # Not degenerate case
#     assert diF.T.conj()!=pytest.approx(diF)
#     assert diF.T!=pytest.approx(diF)
#     assert diF.conj()!=pytest.approx(diF)
# def test_dense_ising_F_complex(dense_ising_F_complex):
#     diF=dense_ising_F_complex[0]
#     assert diF.dtype==np.complex_
#     # Not degenerate case
#     assert diF.conj().T@diF!=pytest.approx(np.eye(diF.shape[0]))
#     assert diF@diF.T.conj()!=pytest.approx(np.eye(diF.shape[0]))
#     assert diF.T.conj()!=pytest.approx(diF)
#     assert diF.T!=pytest.approx(diF)
#     assert diF.conj()!=pytest.approx(diF)
#
# def test_sparse_ising_F_real(dense_ising_F_real):
#     siF=sparse.ising_F(*dense_ising_F_real[1])
#     assert sparse.sparse_to_dense(siF)==pytest.approx(dense_ising_F_real[0])
#
# def test_sparse_ising_F_complex(dense_ising_F_complex):
#     siF=sparse.ising_F(*dense_ising_F_complex[1])
#     assert sparse.sparse_to_dense(siF)==pytest.approx(dense_ising_F_complex[0])
#
# def test_mps_ising_F_complex_obc(dense_ising_F_complex_obc):
#     miF=mps.ising_F(*dense_ising_F_complex_obc[1])
#     assert mps.mpo_to_dense(miF)==pytest.approx(dense_ising_F_complex_obc[0])
