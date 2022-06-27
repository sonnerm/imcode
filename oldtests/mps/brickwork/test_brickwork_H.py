# import imcode.dense as dense
# import imcode.mps as mps
# import pytest
# import numpy as np
#
# def test_two_site_mps_brickwork_H(seed_rng):
#     gate = np.random.normal(size=(4,4)) + 1.0j * np.random.normal(size=(4,4))
#     miH=dense.brickwork.brickwork_H(2,[gate])
#     assert miH.to_dense()==pytest.approx(gate.reshape((4,4)))
#
# def test_four_site_mps_brickwork_H(seed_rng):
#     gates = [np.random.normal(size=(4,4)) + 1.0j * np.random.normal(size=(4,4)) for _ in range(3)]
#     diH=dense.brickwork.brickwork_H(4,gates)
#     miH=dense.brickwork.brickwork_H(4,gates)
#     assert miH.to_dense()==pytest.approx(diH)
