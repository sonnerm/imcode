import numpy as np
from scipy import linalg
np.set_printoptions(suppress=False, linewidth=np.nan)


def entropy(correlation_block, ntimes, time_cut):
    if time_cut == 0:
        # take this as default value if nothing has been specified otherwise
        time_cut = max(ntimes / 2, 1)

    
    """#for correlation approach
    correlation_block_reduced = np.bmat([[correlation_block[0: 4 * time_cut, 0:  4 * time_cut], correlation_block[0: 4 * time_cut, 4 * ntimes: 4 * (time_cut + ntimes)]], [
        correlation_block[4 * ntimes: 4 * (time_cut + ntimes), 0:  4 * time_cut], correlation_block[4 * ntimes: 4 * (time_cut + ntimes), 4 * ntimes: 4 * (time_cut + ntimes)]]])
    """
    
    #for GM-approach
    Delta_half = 2 * time_cut#half the inverval that we define as subsystem
    correlation_block_reduced = np.bmat([[correlation_block[2 * ntimes - Delta_half: 2 * ntimes + Delta_half, 2 * ntimes - Delta_half: 2 * ntimes + Delta_half], correlation_block[2 * ntimes - Delta_half: 2 * ntimes + Delta_half, 6 * ntimes - Delta_half:6 * ntimes + Delta_half]], [
        correlation_block[6 * ntimes - Delta_half: 6 * ntimes + Delta_half,2 * ntimes - Delta_half: 2 * ntimes + Delta_half], correlation_block[6 * ntimes - Delta_half:6 * ntimes + Delta_half, 6 * ntimes - Delta_half: 6 * ntimes + Delta_half]]])
    
    
    
    # print (correlation_block_reduced)
    eigenvalues_correlations, ev_correlations = linalg.eig(correlation_block_reduced)
    eigenvalues_correlations[::-1].sort()
    
    entropy = 0
    for i in range(0, 4 * time_cut, 1):
        kappa = eigenvalues_correlations[i]
        if kappa < 1:  # for kappa = 1, entropy has no contribution
            entropy += - kappa * np.log(kappa) - (1-kappa) * np.log(1-kappa)

    return entropy
