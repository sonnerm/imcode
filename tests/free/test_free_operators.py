# import imcode.free as free
# import imcode.dense as dense
# import scipy.linalg as la
# import imcode.mps.fold as fold
# import pytest
# import numpy as np
#
#
# def test_free_dense_ising_H_L1():
#     seed_rng("free_dense_ising_H_L1")
#     L=1
#     J=np.random.normal(size=(L,))+1.0j*np.random.normal(size=(L,))
#     g=np.random.normal(size=(L,))+1.0j*np.random.normal(size=(L,))
#     J=[0.0]#Just one qubit
#     qtm=free.quad_to_maj(dense.ising_H(J,g,np.zeros_like(J)))
#     assert qtm[0] == pytest.approx(free.ising_H(J,g)[0])
#     assert qtm[1] == pytest.approx(free.ising_H(J,g)[1])
#     assert free.maj_to_quad(free.ising_H(J,g)) == pytest.approx(dense.ising_H(J,g,np.zeros_like(J)))
#
# def test_free_dense_ising_H_L2():
#     seed_rng("free_dense_ising_H_L2")
#     L=2
#     J=np.random.normal(size=(L,))+1.0j*np.random.normal(size=(L,))
#     g=np.random.normal(size=(L,))+1.0j*np.random.normal(size=(L,))
#     J[-1]=0.0#ambigous coupling
#     assert free.maj_to_quad(free.ising_H(J,g)) == pytest.approx(dense.ising_H(J,g,np.zeros_like(J)))
#     qtm=free.quad_to_maj(dense.ising_H(J,g,np.zeros_like(J)))
#     assert qtm[0]/2+qtm[1]/2 == pytest.approx(free.ising_H(J,g)[0]/2+free.ising_H(J,g)[1]/2)#ambiguity
#
# def test_free_dense_ising_H_L5():
#     seed_rng("free_dense_ising_H_L5")
#     L=5
#     J=np.random.normal(size=(L,))+1.0j*np.random.normal(size=(L,))
#     g=np.random.normal(size=(L,))+1.0j*np.random.normal(size=(L,))
#     qtm=free.quad_to_maj(dense.ising_H(J,g,np.zeros_like(J)))
#     assert qtm[0] == pytest.approx(free.ising_H(J,g)[0])
#     assert qtm[1] == pytest.approx(free.ising_H(J,g)[1])
#     assert free.maj_to_quad(free.ising_H(J,g)) == pytest.approx(dense.ising_H(J,g,np.zeros_like(J)))
#
# def test_free_dense_ising_F_obc_L1():
#     seed_rng("free_dense_ising_F_obc_L1")
#     L=1
#     J=np.random.normal(size=(L,))+1.0j*np.random.normal(size=(L,))
#     g=np.random.normal(size=(L,))+1.0j*np.random.normal(size=(L,))
#     J=[0.0]
#     ttm=free.simple_to_maj(dense.ising_F(J,g,np.zeros_like(J)))
#     print(la.logm(ttm))
#     print(la.logm(free.ising_F(J,g)[0]))
#     assert ttm == pytest.approx(free.ising_F(J,g)[0])
#
# def test_free_dense_ising_F_obc_L2():
#     seed_rng("free_dense_ising_F_obc_L2")
#     L=2
#     J=np.random.normal(size=(L,))+1.0j*np.random.normal(size=(L,))
#     g=np.random.normal(size=(L,))+1.0j*np.random.normal(size=(L,))
#     J[-1]=0.0
#     ttm=free.simple_to_maj(dense.ising_F(J,g,np.zeros_like(J)))
#     print(ttm)
#     print(free.ising_F(J,g)[0])
#     assert ttm == pytest.approx(free.ising_F(J,g)[0])
#
# def test_free_dense_ising_F_obc_L5():
#     seed_rng("free_dense_ising_F_obc_L5")
#     L=5
#     J=np.random.normal(size=(L,))+1.0j*np.random.normal(size=(L,))
#     g=np.random.normal(size=(L,))+1.0j*np.random.normal(size=(L,))
#     J[-1]=0.0
#     ttm=free.simple_to_maj(dense.ising_F(J,g,np.zeros_like(J)))
#     assert ttm == pytest.approx(free.ising_F(J,g)[0],1e-9,1e-9)
# @pytest.mark.skip()
# def test_free_dense_ising_J_T1():
#     seed_rng("free_dense_ising_J_T1")
#     T=1
#     J=np.random.normal()
#     ttm=free.simple_to_maj(dense.ising_J(T,J))
#     assert ttm == pytest.approx(free.ising_J(T,J)[0],1e-9,1e-9)
#
# @pytest.mark.skip()
# def test_free_dense_ising_J_T2():
#     seed_rng("free_dense_ising_J_T2")
#     T=2
#     J=np.random.normal()
#     ttm=free.simple_to_maj(dense.ising_J(T,J))
#     assert ttm == pytest.approx(free.ising_J(T,J)[0],1e-9,1e-9)
#
# @pytest.mark.skip()
# def test_free_dense_ising_J_T3():
#     seed_rng("free_dense_ising_J_T3")
#     T=3
#     J=np.random.normal()
#     ttm=free.simple_to_maj(dense.ising_J(T,J))
#     assert ttm == pytest.approx(free.ising_J(T,J)[0],1e-9,1e-9)
