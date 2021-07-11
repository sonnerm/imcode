import numpy as np
from scipy.linalg import expm
from scipy import linalg

def compute_generators(nsites, Jx=0, Jy=0, g=0):
    # define generators for unitary transformation

    # G_XY - two-site gates (XX + YY)
    G_XY_odd = np.zeros((2 * nsites, 2 * nsites))
    G_XY_even = np.zeros((2 * nsites, 2 * nsites))

    Jp = (Jx + Jy)
    Jm = (Jy - Jx)

    if abs(Jm) < 1e-10:
        Jm = 1e-10
    if abs(g) < 1e-10:
        g = 1e-10

    eps = 1e-8  # lift degeneracy
    G_XY_odd[0, nsites - 1] += eps
    G_XY_odd[nsites - 1, 0] += eps
    G_XY_odd[nsites, 2 * nsites - 1] += -eps
    G_XY_odd[2 * nsites - 1, nsites] += -eps

    G_XY_odd[nsites - 1, nsites] -= eps
    G_XY_odd[0, 2 * nsites - 1] += eps
    G_XY_odd[2 * nsites - 1, 0] += eps
    G_XY_odd[nsites, nsites - 1] -= eps

    for i in range(0, nsites - 1, 2):
        G_XY_even[i, i + 1] = Jp
        G_XY_even[i + 1, i] = Jp
        G_XY_even[i, i + nsites + 1] = -Jm
        G_XY_even[i + 1, i + nsites] = Jm
        G_XY_even[i + nsites, i + 1] = Jm
        G_XY_even[i + nsites + 1, i] = -Jm
        G_XY_even[i + nsites, i + nsites + 1] = -Jp
        G_XY_even[i + nsites + 1, i + nsites] = -Jp

    for i in range(1, nsites - 1, 2):
        G_XY_odd[i, i + 1] = Jp
        G_XY_odd[i + 1, i] = Jp
        G_XY_odd[i, i + nsites + 1] = -Jm
        G_XY_odd[i + 1, i + nsites] = Jm
        G_XY_odd[i + nsites, i + 1] = Jm
        G_XY_odd[i + nsites + 1, i] = -Jm
        G_XY_odd[i + nsites, i + nsites + 1] = - Jp
        G_XY_odd[i + nsites + 1, i + nsites] = - Jp

    # G_g - single body kicks
    G_g = np.zeros((2 * nsites, 2 * nsites))
    for i in range(nsites):
        G_g[i, i] = - 2 * g
        G_g[i + nsites, i + nsites] = 2 * g

    # G_1 - residual gate coming from projecting interaction gate of xy-model on the vacuum at site 0
    G_1 = np.zeros((2 * nsites, 2 * nsites))

    beta_tilde = np.arctanh(np.tan(Jx) * np.tan(Jy))

    G_1[0, 0] = 2 * beta_tilde
    G_1[nsites, nsites] = -2 * beta_tilde

    # give out explicit form of generators
    print('G_XY_even = ')
    print(G_XY_even)

    print('G_XY_odd = ')
    print(G_XY_odd)

    print('G_g = ')
    print(G_g)

    print('G_1 = ')
    print(G_1)

    return G_XY_even, G_XY_odd, G_g, G_1