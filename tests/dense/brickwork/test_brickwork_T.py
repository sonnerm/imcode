import numpy as np
import numpy.linalg as la
from imcode import dense
import pytest
# def test_brickwork_Sa(seed_rng):
#     T=3
#     h=np.random.normal()
#     dih=dense.ising.ising_h(T,h)
#     assert dih.dtype==np.complex_
#     assert dih.shape==(64,64)
#     assert np.diag(np.diag(dih))==pytest.approx(dih) #diagonal
#     assert dih.conj()*dih==pytest.approx(np.eye(dih.shape[0])) #unitary
#     assert dih.conj()!=pytest.approx(dih)
# def test_dense_ising_h_zero():
#     T=3
#     h=0.0
#     dih=dense.ising.ising_h(T,h)
#     assert dih==pytest.approx(np.eye(64))
#
# def test_dense_ising_h_complex(seed_rng):
#     T=3
#     h=np.random.normal()+np.random.normal()*1.0j
#     dih=dense.ising.ising_h(T,h)
#     assert dih.dtype==np.complex_
#     assert dih.shape==(64,64)
#     assert np.diag(np.diag(dih))==pytest.approx(dih) #diagonal
#     assert dih.conj()*dih!=pytest.approx(np.eye(dih.shape[0]))
#     assert dih.conj()!=pytest.approx(dih)
#
# def test_dense_ising_g_real(seed_rng):
#     T=3
#     g=np.random.normal()
#     init=np.random.normal(size=(2,2))+1.0j*np.random.normal(size=(2,2))
#     final=np.random.normal(size=(2,2))+1.0j*np.random.normal(size=(2,2))
#     init=init.T.conj()+init #init hermitian
#     final=final.T.conj()+final #final hermitian
#     diW=dense.ising.ising_g(T,g,init,final)
#     assert diW.dtype==np.complex_
#     assert diW.shape==(64,64)
#     assert np.diag(np.diag(diW))==pytest.approx(diW) #diagonal
#     assert diW.conj()*diW!=pytest.approx(np.eye(64)) #generic
#     assert diW.conj()!=pytest.approx(diW)
#
# def test_dense_ising_g_complex(seed_rng):
#     T=3
#     g=np.random.normal()+1.0j*np.random.normal()
#     init=np.random.normal(size=(2,2))+1.0j*np.random.normal(size=(2,2))
#     final=np.random.normal(size=(2,2))+1.0j*np.random.normal(size=(2,2))
#     diW=dense.ising.ising_g(T,g,init,final)
#     assert diW.dtype==np.complex_
#     assert diW.shape==(64,64)
#     assert np.diag(np.diag(diW))==pytest.approx(diW) #diagonal
#     assert diW.conj()*diW!=pytest.approx(np.eye(diW.shape[0]))
#     assert diW.conj()!=pytest.approx(diW)
# def test_dense_ising_g_T1(seed_rng):
#     T=1
#     g=np.random.normal()
#     diW=dense.ising.ising_g(T,g)
#
# def test_dense_ising_g_zero():
#     T=3
#     g=0.0
#     diW=dense.ising.ising_g(T,g)
#     diWc=np.zeros((4**T,4**T))
#     diWc[0,0]=0.5
#     diWc[-1,-1]=0.5
#     assert diW==pytest.approx(diWc)
#
# # @pytest.mark.skip
# # def test_dense_ising_g_pi4():
# #     #dual unitary case, not 100% sure what to expect
# #     T=2
# #     g=np.pi/4
# #     diW=dense.ising.ising_g(T,g,init=np.ones((2,2)),final=np.ones((2,2)))*4
# #     print(np.diag(diW))
# #     print(np.diag(diW.conj())*np.diag(diW))
# #     assert diW.dtype==np.complex_
# #     assert diW.shape==(4**T,4**T)
# #     assert np.diag(np.diag(diW))==pytest.approx(diW) #diagonal
# #     assert diW.conj()*diW==pytest.approx(np.eye(diW.shape[0])) #unitary
# #     assert diW.conj()!=pytest.approx(diW)
#
# def test_dense_ising_J_complex(seed_rng):
#     T=3
#     J=np.random.normal()+1.0j*np.random.normal()
#     diJ=dense.ising.ising_J(T,J)
#     assert diJ.dtype==np.complex_
#     assert diJ.shape==(64,64)
#     assert diJ.conj()*diJ!=pytest.approx(np.eye(diJ.shape[0]))
#     assert diJ.conj()!=pytest.approx(diJ)
#     #TODO: test separable
# def test_dense_ising_J_real(seed_rng):
#     T=3
#     J=np.random.normal()
#     diJ=dense.ising.ising_J(T,J)
#     assert diJ.dtype==np.complex_
#     assert diJ.shape==(64,64)
#     assert diJ.conj()*diJ!=pytest.approx(np.eye(diJ.shape[0]))
#     assert diJ.conj()!=pytest.approx(diJ)
# def test_dense_ising_J_zero(seed_rng):
#     T=3
#     J=0.0
#     diJ=dense.ising.ising_J(T,J)
#     assert diJ==pytest.approx(np.ones((64,64)))
#
# def test_dense_ising_T_real(seed_rng):
#     T=4
#     J=np.random.normal()
#     g=np.random.normal()
#     h=np.random.normal()
#     init=np.random.normal(size=(2,2))+1.0j*np.random.normal(size=(2,2))
#     init=init.T.conj()+init
#     init=init@init
#     init/=np.trace(init)#init is a proper density matrix
#     init=np.array([[1,0],[0,0]])
#     diT=dense.ising.ising_T(T,J,g,h,init)
#     ch=dense.unitary_channel(dense.ising.ising_F(1,[],[g],[h]))
#     diT2=dense.ising.ising_J(T,J)@dense.ising.ising_W(T,[ch,ch,ch,ch],init)
#     # print(np.where(np.abs(diT-diT2)>1e-6))
#     # print('=======')
#     # print(diT[np.where(np.abs(diT-diT2)>1e-6)])
#     # print(diT2[np.where(np.abs(diT-diT2)>1e-6)])
#     assert diT==pytest.approx(diT2,abs=1e-6,rel=1e-6)
#     assert diT.dtype==np.complex_
#     assert diT.shape==(4**T,4**T)
#     ditev=la.eigvals(diT)
#     ditevc=np.zeros((4**T))
#     ditevc[-1]=1.0
#     assert np.sort_complex(ditev)==pytest.approx(ditevc,rel=5e-4,abs=5e-4)
#
#
# def test_dense_ising_T_pi4():
#     T=3
#     J=np.pi/4
#     g=np.pi/4
#     h=1.0
#     # init=np.random.normal(size=(2,2))+1.0j*np.random.normal(size=(2,2))
#     # init=init.T.conj()+init
#     # init=init@init
#     # init/=np.trace(init)#init is a proper density matrix
#     init=np.eye(2)/2
#     diT=dense.ising.ising_T(T,J,g,h,init)
#     ch=dense.unitary_channel(dense.ising.ising_F(1,[],[g],[h]))
#     diT2=dense.ising.ising_J(T,J)@dense.ising.ising_W(T,[ch]*T)
#     assert diT==pytest.approx(diT2)
#     assert diT.dtype==np.complex_
#     assert diT.shape==(64,64)
#     ditev,ditevv=la.eig(diT)
#     ditevc=np.zeros((4**T))
#     ditevc[-1]=1.0
#     pdev=ditevv[:,np.argmax(np.abs(ditev))]
#     print(pdev/pdev[0])
#     assert np.sort_complex(ditev)==pytest.approx(ditevc,rel=5e-4,abs=5e-4)
#     assert pdev/pdev[0]==pytest.approx(dense.ising.perfect_dephaser_im(T),rel=1e-7,abs=1e-7)
# def test_dense_ising_W(seed_rng):
#     T=3
#     g=np.random.normal()+1.0j*np.random.normal()
#     h=np.random.normal()+1.0j*np.random.normal()
#     ch=dense.unitary_channel(dense.ising.ising_F(1,[],[g],[h]))
