import numpy as np


def correlation_coefficients(T_tilde):

    # initialize correlation coefficients
    # first dimension 2 is for forward/backward branch (0 is forward, 1 is backward), second dimension 2 is for "Majorana" type (0 is THETA (relative MINUS sign), 1 is ZETA (relative PLUS sign)), third axis is for evolution times, fourth axis is for site arguments: (A_ij)-> site argument i=0 always, j in range (0,2L)
    A = np.zeros((2, 2, ntimes, 2*nsites), dtype=np.complex_)

    # fill with values (individually for each branch)
    for tau in range(0, ntimes):
        # buffer evolver values that are combined to give correlation coefficients A
        # .. on forward branch
        # arguments:forward branch, evolution time, site j=0 in env.
        T_tilde_fw_0 = T_tilde[0, tau, 0]
        # arguments:forward branch, evolution time, site j=0+L in env.
        T_tilde_fw_L = T_tilde[0, tau, 1]
        # .. on backward branch
        # arguments:forward branch, evolution time, site j=0 in env.
        T_tilde_bw_0 = T_tilde[1, tau, 0]
        # arguments:forward branch, evolution time, site j=0+L in env.
        T_tilde_bw_L = T_tilde[1, tau, 1]

        # fill array of correlation coefficients
        # arguments: forward branch, "Theta/- Majorana"
        A[0, 0] = T_tilde_fw_0 - T_tilde_fw_L
        # arguments: forward branch, "Zeta/+ Majorana"
        A[0, 1] = T_tilde_fw_0 + T_tilde_fw_L
        # arguments: backward branch, "Theta/- Majorana"
        A[1, 0] = T_tilde_bw_0 - T_tilde_bw_L
        # arguments: backward branch, "Zeta/+ Majorana"
        A[1, 1] = T_tilde_bw_0 + T_tilde_bw_L

    return A
