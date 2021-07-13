import imcode.mps as mps
import numpy as np
import imcode.dense as dense
import pytest
def test_im_rectangle_dense(seed_rng):
    J,g,h=np.random.normal(size=3)
    t=5
    T=mps.ising.ising_T(t,J,g,h)
    ims=[im for im in mps.ising.im_rectangle(T,chi=64)]
    assert ims[-1].to_dense()==pytest.approx(ims[-2].to_dense()) #convergence achieved
    assert ims[-1].to_dense() == pytest.approx(dense.ising.im_diag(T.to_dense())) # converged to the right ev

def test_im_rectangle(seed_rng):
    J,g,h=np.random.normal(size=3)
    t=10
    T=mps.ising.ising_T(t,J,g,h)
    ims=[im for im in mps.ising.im_rectangle(T,chi=64)]
    assert ims[-1].conj()@ims[-2]==pytest.approx(np.sqrt((ims[-1].conj()@ims[-1])*(ims[-2].conj()@ims[-2]))) #convergence achieved

# def test_im_triangle(seed_rng):
#     J,g,h=np.random.normal(size=3)
#     t=10
#     Ts=[mps.ising.ising_T(tt,J,g,h) for tt in range(1,t+1)]
#     ims=[im for im in mps.ising.im_triangle(Ts,chi=64)]
#     imc=[list(mps.ising.im_rectangle(T,chi=64))[-1] for T in Ts]
def test_im_diamond(seed_rng):
    J,g,h=np.random.normal(size=3)
    t=10
    Ts=[mps.ising.ising_T(tt,J,g,h) for tt in range(1,t+1,2)]
    ims=[im for im in mps.ising.im_diamond(Ts,chi=64)]
    imc=[list(mps.ising.im_diamond(T,chi=64))[-1] for T in Ts]
