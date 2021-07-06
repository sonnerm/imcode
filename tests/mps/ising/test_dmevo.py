import imcode.mps as mps
import scipy.linalg as la
import numpy as np

def test_boundary_single_dmevo(seed_rng):
    L=4
    t=4
    chi=64
    Js=np.random.normal(size=(L,))
    gs=np.random.normal(size=(L,))
    hs=np.random.normal(size=(L,))
    Ts=[mps.ising.ising_T(t,J,g,h,np.eye(2),np.eye(2)) for J,g,h in zip(Js,gs,hs)]
    lops=[np.random.normal(size=(2,2))+np.random.normal(size=(2,2))*1.0j for _ in range(t)]
    lops=[la.expm(l-l.T.conj()) for l in lops]
    init=np.random.normal(size=(2,2))+np.random.normal(size=(2,2))*1.0j
    init=init+init.T.conj()
    im=mps.ising.open_boundary_im(t)
    for T in Ts:
        im=(T@im).contract(chi_max=chi)
    dms=mps.ising.boundary_dm_evolution(im,lops,init)

def test_embedded_double_dmevo(seed_rng):
    pass
