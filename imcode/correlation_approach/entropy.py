import numpy as np
from scipy.linalg import expm, schur, eigvals
from scipy import linalg
from scipy.sparse.linalg import eigsh
np.set_printoptions(suppress=False, linewidth=np.nan)


def entropy(correlation_block, ntimes, time_cut):
    if time_cut == 0:
        # take this as default value if nothing has been specified otherwise
        time_cut = max(ntimes / 2, 1)

    correlation_block_reduced = np.bmat([[correlation_block[0: 4 * time_cut, 0:  4 * time_cut], correlation_block[0: 4 * time_cut, 4 * ntimes: 4 * (time_cut + ntimes)]], [
        correlation_block[4 * ntimes: 4 * (time_cut + ntimes), 0:  4 * time_cut], correlation_block[4 * ntimes: 4 * (time_cut + ntimes), 4 * ntimes: 4 * (time_cut + ntimes)]]])
    # print correlation_block_reduced
    eigenvalues_correlations, ev_correlations = eigsh(
        correlation_block_reduced, 8 * time_cut)
    eigenvalues_correlations[::-1].sort()
    # print 'cut:' , cut , ', reduced eigenvalue correlations:',eigenvalues_correlations

    entropy = 0
    for i in range(0, 4 * time_cut, 1):
        kappa = eigenvalues_correlations[i]
        if kappa < 1:  # for kappa = 1, entropy has no contribution
            entropy += - kappa * np.log(kappa) - (1-kappa) * np.log(1-kappa)

    return entropy
