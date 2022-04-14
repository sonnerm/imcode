import numpy as np
from scipy import linalg

np.set_printoptions(suppress=False, linewidth=np.nan)


def entropy(mode, correlation_block, ntimes, time_cut, iterator,filename):

    half_dim_CB = correlation_block.shape[0] // 2

    if time_cut == 0:
        # take this as default value if nothing has been specified otherwise
        time_cut = max(ntimes / 2, 1)


    if mode == 'C' or mode == 'D':#for "C"orrelation approach and "D"MFT data evaluation
        Delta_half = 2. * time_cut / (8. * ntimes / correlation_block.shape[0])#half the inverval that we define as subsystem    
        correlation_block_reduced = np.bmat([[correlation_block[0: int(2 * Delta_half), 0:  int(2 * Delta_half)], correlation_block[0: int(2 * Delta_half), half_dim_CB: half_dim_CB + int(2 * Delta_half)]], [
            correlation_block[half_dim_CB: half_dim_CB + int(2 * Delta_half), 0:  int(2 * Delta_half)], correlation_block[half_dim_CB: half_dim_CB + int(2 * Delta_half), half_dim_CB: half_dim_CB + int(2 * Delta_half)]]])
        #if time_cut < 2:
            #print(correlation_block_reduced.shape)
            #print(correlation_block_reduced)
    else:#for Grassmann-approach
        Delta_half = 2 * time_cut
        correlation_block_reduced = np.bmat([[correlation_block[half_dim_CB // 2 - Delta_half: half_dim_CB // 2 + Delta_half, half_dim_CB // 2 - Delta_half: half_dim_CB // 2 + Delta_half], correlation_block[half_dim_CB // 2 - Delta_half: half_dim_CB // 2 + Delta_half, 3 * (half_dim_CB // 2) - Delta_half:3 * (half_dim_CB // 2) + Delta_half]], [
            correlation_block[3 * (half_dim_CB // 2) - Delta_half: 3 * (half_dim_CB // 2) + Delta_half, (half_dim_CB // 2) - Delta_half: (half_dim_CB // 2) + Delta_half], correlation_block[3 * (half_dim_CB // 2) - Delta_half:3 * (half_dim_CB // 2) + Delta_half, 3 * (half_dim_CB // 2) - Delta_half: 3 * (half_dim_CB // 2) + Delta_half]]])
    


    eigenvalues_correlations, ev_correlations = linalg.eigh(correlation_block_reduced)
    #print(eigenvalues_correlations)
    #print(ev_correlations)
    eigenvalues_correlations[::-1].sort()

    #print(ev_correlations.T.conj() @ correlation_block_reduced @ ev_correlations)
    np.set_printoptions(linewidth=np.nan, precision=2, suppress=True)
    #print('time cut', time_cut)
    #print('eigenvalues_correlations')
    #print(eigenvalues_correlations)
    with h5py.File(filename + '.hdf5', 'a') as f:
            data = f['entangl_spectr']
            data[iterator, time_cut,0:len(eigenvalues_correlations)] = np.array(np.real(eigenvalues_correlations))

    entropy = 0
    for i in range(correlation_block_reduced.shape[0] // 2):
        kappa = eigenvalues_correlations[i]
        if kappa < 1:  # for kappa = 1, entropy has no contribution
            entropy += - kappa * np.log(kappa) - (1-kappa) * np.log(1-kappa)

    return entropy
