# import imcode.mps as mps
# from tenpy.linalg.charges import LegCharge
# from tenpy.networks.mpo import MPO
# import tenpy.linalg.np_conserved as npc
# import imcode.mps.flat as flat
# import pytest
#
# import numpy as np
# def test_mps_direct_dm_evo():
#     seed_rng("mps_direct_em_evo")
#     L=8
#     chi=64
#     t=6
#     J=np.random.normal()
#     g=np.random.normal()
#     h=np.random.normal()
#     T=mps.fold.ising_T(t,J,g,h)
#     im=mps.im_iterative(T)
#     lop=mps.multiply_mpos([mps.fold.ising_W(t,g),mps.fold.ising_h(t,h)])
#     F=mps.ising_F([J]*(L-1),[g]*L,[h]*L)
#     dms=mps.dm_evolution(im,lop,[1,0,0,0])
#     dmsd=mps.direct_dm_evolution(F,0,t,chi,linit=[[1,0],[0,0]])
#     for d1,d2 in zip(dms,dmsd):
#         print("d1:")
#         print(d1)
#         print("d2:")
#         print(d2)
#         assert d1==pytest.approx(d2)
