import numpy as np
def ising_gamma(M, eigenvalues_G_eff, nsites):#computes response function gamma that appears in the Ising chain
    M_fw = M[0]
    eigenvalues_G_eff_fw = eigenvalues_G_eff[0]
    gamma_test_range = 100
    gamma_test_vals = np.zeros(gamma_test_range)
    for i in range(gamma_test_range):
        gamma_test = 0
        for k in range(nsites):
            gamma_test += abs(M_fw[0, k] - M_fw[nsites, k])**2 * np.cos(eigenvalues_G_eff_fw[k] * i)
        gamma_test_vals[i] = gamma_test
    return np.arange(0, gamma_test_range), gamma_test_vals