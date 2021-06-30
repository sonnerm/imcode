import numpy as np
from numpy.core.einsumfunc import einsum
from numpy.linalg import matrix_power
np.set_printoptions(suppress=False, linewidth=np.nan)


def evolvers(M_fw, M_fw_inverse, M_bw, M_bw_inverse,  N_t, eigenvalues_G_eff_fw, eigenvalues_G_eff_bw, nsites, ntimes, beta_tilde):

    # initialize diagonal matrices that encode..
    # .. time evolution ..
    D_phi_fw = np.zeros((2*nsites, 2*nsites), dtype=np.complex_)
    D_phi_bw = np.zeros((2*nsites, 2*nsites), dtype=np.complex_)
    # .. and dressing
    D_beta_fw = np.zeros((2, 2*nsites), dtype=np.complex_)
    D_beta_bw = np.zeros((2, 2*nsites), dtype=np.complex_)

    # fill with values (individually for each branch)
    for i in range(0, nsites):
        # forward branch
        D_phi_fw[i, i] = np.exp(-1j*eigenvalues_G_eff_fw[i])
        D_phi_fw[i+nsites, i +
                 nsites] = np.exp(-1j*eigenvalues_G_eff_fw[i+nsites])

        # backward branch
        D_phi_bw[i, i] = np.exp(-1j*eigenvalues_G_eff_bw[i])
        D_phi_bw[i+nsites, i +
                 nsites] = np.exp(-1j*eigenvalues_G_eff_bw[i+nsites])

    D_beta_fw[0, 0] = np.exp(beta_tilde)
    D_beta_fw[1, nsites] = np.exp(-beta_tilde)
    D_beta_bw[0, 0] = np.exp(-beta_tilde)
    D_beta_bw[1, nsites] = np.exp(beta_tilde)

    # initialize evolvers
    # first dimension 2 is for forward/backward branch (0 is forward, 1 is backward), second dimension 2 is because we do not compute correlations at every site in the environment but only at site 1 -> two relevant lines (site 0 and 0+L in env.) are extracted (in principle one would have dimension 2*nsites here,too)
    T_tilde = np.zeros((2, ntimes, 2, 2*nsites), dtype=np.complex_)
    #T_tilde_c = np.zeros((2, ntimes, 2, 2*nsites), dtype=np.complex_)

    # fill with values (individually for each branch)
    #branch, tau = np.mgrid[0:2,0:ntimes]
    for tau in range(0, ntimes):
        T_tilde[0, tau] = np.einsum('ij,jk,kl,lm,mn->in', D_beta_fw, M_fw,matrix_power(D_phi_fw, tau + 1), M_fw_inverse, N_t)  # forward branch
        T_tilde[1, tau] = np.einsum('ij,jk,kl,lm,mn->in', D_beta_bw, M_bw,matrix_power(D_phi_bw, tau + 1), M_bw_inverse, N_t)  # backward branch
    #T_tilde_c[branch,tau] = np.einsum('ij,jk,kl,lm,mn->in', D_beta_fw, M_fw,matrix_power(D_phi_fw, tau + 1), M_bw_inverse, N_t)  # forward branch

    return T_tilde
