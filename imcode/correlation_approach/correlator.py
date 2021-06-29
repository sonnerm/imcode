import numpy as np
from numpy import version
import matplotlib.pyplot as plt
np.set_printoptions(suppress=False, linewidth=np.nan)


# i and j are site indices, s and sp specify whether the fermionic operators have a dagger (=1) or not (=0), t2 and t1 denote times, M is the matrix of eigenvetors (as columns) and eigenvalues_G_eff contains eigenvalues of G_eff )
# returns greater correlation function
def correlator(M, eigenvalues_G_eff, s, sp, i, j, t2, t1, beta=0):
    result = 0
    nsites = int(eigenvalues_G_eff.size / 2)
    if beta > 0:
        for k in range(2 * nsites):
            for kp in range(2 * nsites):
                for l in range(0, nsites):
                    result += M[j + sp * nsites, kp] * M[i + s * nsites, k] * np.exp(-1j * (
                        eigenvalues_G_eff[kp] * t2 + eigenvalues_G_eff[k] * t1)) * (M.conj().T[k, l] * M.conj().T[kp, l + nsites] * 1/(1 + np.exp(beta)) + M.conj().T[k, l + nsites] * M.conj().T[kp, l] * 1/(1 + np.exp(-beta)))
    else:
        for k in range(nsites):

            result += 0.5 * M[j + (1 - sp) * nsites, k].conj() * M[i + s * nsites, k] * np.exp(1j * eigenvalues_G_eff[k] * (
                t2 - t1)) + 0.5 * M[j + sp * nsites, k] * M[i + (1-s) * nsites, k].conj() * np.exp(-1j * eigenvalues_G_eff[k] * (t2 - t1))
    return result

