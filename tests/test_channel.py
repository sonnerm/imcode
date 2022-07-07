import numpy as np
import ttarray as tt
import pytest
import numpy.linalg as la
import imcode
def test_unitary_channel_L1(seed_rng):
    H=np.random.normal(size=(2,2))+1.0j*np.random.normal(size=(2,2))
    H=H+H.T.conj()
    U=la.eigh(H)[1]
    rho=np.random.normal(size=(2,2))+1.0j*np.random.normal(size=(2,2))
    rho=rho@rho.T.conj()
    rho/=np.trace(rho)
    rhof1=U@rho@U.T.conj()
    Ch=imcode.unitary_channel(U)
    rhof2=imcode.unvectorize_operator(Ch@imcode.vectorize_operator(rho))
    assert rhof2==pytest.approx(rhof1)


def test_unitary_channel_L4(seed_rng):
    L=4
    H=np.random.normal(size=(2**L,2**L))+1.0j*np.random.normal(size=(2**L,2**L))
    H=H+H.T.conj()
    U=la.eigh(H)[1]
    rho=np.random.normal(size=(2**L,2**L))+1.0j*np.random.normal(size=(2**L,2**L))
    rho=rho@rho.T.conj()
    rho/=np.trace(rho)
    rhof1=U@rho@U.T.conj()
    Ch=imcode.unitary_channel(U)
    rhof2=imcode.unvectorize_operator(Ch@imcode.vectorize_operator(rho))
    assert rhof2==pytest.approx(rhof1)

# def test_unitary_channel_slice(seed_rng):
#     L=4
#     H=np.random.normal(size=(2**L,2**L))+1.0j*np.random.normal(size=(2**L,2**L))
#     H=H+H.T.conj()
#     U=la.eigh(H)[1]
#     rho=np.random.normal(size=(2**L,2**L))+1.0j*np.random.normal(size=(2**L,2**L))
#     rho=rho@rho.T.conj()
#     rho/=np.trace(rho)
#     rhos=tt.array(rho).M[1:2]
#     Us=tt.array(U).M[1:2]
#     rhosf1=Us@rhos@Us.T.conj()
#     Ch=imcode.unitary_channel(Us)
#     rhosf2=imcode.unvectorize_operator(Ch@imcode.vectorize_operator(rhos))
#     assert rhosf2==pytest.approx(rhosf1)
