# import imcode.free as free
# import imcode.dense as dense
#
# import numpy as np
# import pytest
# @pytest.mark.skip()
# def test_free_direct_czz_open():
#     L=5
#     seed_rng("free_direct_czz_open")
#     J=np.random.normal(size=(L,))+1.0j*np.random.normal(size=(L,))
#     g=np.random.normal(size=(L,))+1.0j*np.random.normal(size=(L,))
#     J[-1]=0.0
#     dF=dense.ising_F(J,g,np.zeros_like(g))
#     fF=free.ising_F(J,g)
#     for i in range(L):
#         for j in range(L):
#             for t in range(5):
#                 assert free.direct_czz(fF,i,j,t) == pytest.approx(dF,i,j,t)
