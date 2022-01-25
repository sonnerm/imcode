import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
import h5py
np.set_printoptions(suppress=False, linewidth=np.nan)


def entropy(mode, correlation_block, ntimes, time_cut, iterator,filename):

    half_dim_CB = correlation_block.shape[0] // 2

    if time_cut == 0:
        # take this as default value if nothing has been specified otherwise
        time_cut = max(ntimes / 2, 1)


    if mode == 'C' or mode == 'D':#for "C"orrelation approach and "D"MFT data evaluation
        Delta_half = 2. * time_cut / (8. * ntimes / correlation_block.shape[0])#half the inverval that we define as subsystem    
        print(Delta_half)
        correlation_block_reduced = np.bmat([[correlation_block[0: int(2 * Delta_half), 0:  int(2 * Delta_half)], correlation_block[0: int(2 * Delta_half), half_dim_CB: half_dim_CB + int(2 * Delta_half)]], [
            correlation_block[half_dim_CB: half_dim_CB + int(2 * Delta_half), 0:  int(2 * Delta_half)], correlation_block[half_dim_CB: half_dim_CB + int(2 * Delta_half), half_dim_CB: half_dim_CB + int(2 * Delta_half)]]])
    
    else:#for Grassmann-approach
        Delta_half = 2 * time_cut
        correlation_block_reduced = np.bmat([[correlation_block[half_dim_CB // 2 - Delta_half: half_dim_CB // 2 + Delta_half, half_dim_CB // 2 - Delta_half: half_dim_CB // 2 + Delta_half], correlation_block[half_dim_CB // 2 - Delta_half: half_dim_CB // 2 + Delta_half, 3 * (half_dim_CB // 2) - Delta_half:3 * (half_dim_CB // 2) + Delta_half]], [
            correlation_block[3 * (half_dim_CB // 2) - Delta_half: 3 * (half_dim_CB // 2) + Delta_half, (half_dim_CB // 2) - Delta_half: (half_dim_CB // 2) + Delta_half], correlation_block[3 * (half_dim_CB // 2) - Delta_half:3 * (half_dim_CB // 2) + Delta_half, 3 * (half_dim_CB // 2) - Delta_half: 3 * (half_dim_CB // 2) + Delta_half]]])
    

    
    """
    print ('sub')
    sub=correlation_block[0: int(2 * Delta_half), 0:  int(2 * Delta_half)]
    print(sub)
    eigenvalues_correlations, ev_correlations = linalg.eig(sub)
    print(eigenvalues_correlations)
    print(ev_correlations)

    print ('sub correlation_block_reduced')
    """
    #correlation_block_reduced -= 0.5*np.identity(correlation_block_reduced.shape[0])

    """
    T = np.zeros(correlation_block.shape)
    for i in range (correlation_block.shape[0] // 4):
        T [correlation_block.shape[0] // 2 - (2 * i) - 2,4 * i] = 1
        T [correlation_block.shape[0] // 2 - (2 * i) - 1,4 * i + 2] = 1
        T [correlation_block.shape[0] // 2 + (2 * i) ,4 * i + 1] = 1
        T [correlation_block.shape[0] // 2 + (2 * i) + 1,4 * i + 3] = 1
    correlation_block = T @ correlation_block @ T.T

    S = np.zeros(correlation_block_reduced.shape)
    for i in range (correlation_block_reduced.shape[0] // 4):
        S [correlation_block_reduced.shape[0] // 2 - (2 * i) - 2,4 * i] = 1
        S [correlation_block_reduced.shape[0] // 2 - (2 * i) - 1,4 * i + 2] = 1
        S [correlation_block_reduced.shape[0] // 2 + (2 * i) ,4 * i + 1] = 1
        S [correlation_block_reduced.shape[0] // 2 + (2 * i) + 1,4 * i + 3] = 1
    correlation_block_reduced = S @ correlation_block_reduced @ S.T
    """    
    """
    rot = np.zeros(correlation_block_reduced.shape, dtype=np.complex_)
    for i in range(0,correlation_block_reduced.shape[0], 2):
        rot[i,i] = 1./np.sqrt(2)
        rot[i,i+1] = 1./np.sqrt(2)
        rot[i+1,i] =  -1.j/np.sqrt(2) #* np.sign(correlation_block_reduced.shape[0]//2 - i-1)
        rot[i+1,i+1] = 1.j/np.sqrt(2) #* np.sign(correlation_block_reduced.shape[0]//2 - i-1)
    
    correlation_block_reduced = rot @ correlation_block_reduced @ rot.conj().T
    """    
    #print(correlation_block)
    #print(correlation_block_reduced)
    #fig, ax = plt.subplots()


    #ax.matshow(abs(correlation_block))
    #plt.show()
    

    """
    #print(sub_corr)
    eigenvalues_correlations, ev_correlations = linalg.eig(sub_corr)
    argsort = np.argsort(eigenvalues_correlations)
    R = np.zeros(ev_correlations.shape)
    eigenvalues_correlations[::-1].sort()
    for i in range(ev_correlations.shape[0]):  # sort eigenvectors and eigenvalues such that the first half are the ones with positive real part, and the second half have negative real parts
        R[:, i] = ev_correlations[:, argsort[i]]

    if time_cut == 4 and ntimes == 9:
        #R[:,[0, 4]] = R[:,[4, 0]]
        #R[:,[1, 5]] = R[:,[5, 1]]

        fig, ax = plt.subplots(1)
        x = []
        y = []
        for i in range (10,12):
            x = np.arange(ev_correlations.shape[0])
            y = ((R[:,i]))
            ax.plot(x, y)
    
    plt.show()
    print('eigenvalues_correlations red')
    #print(eigenvalues_correlations)
    #print(ev_correlations)
    #print(ev_correlations.T @ sub_corr @ ev_correlations)

    angle = -np.arctan(ev_correlations[ev_correlations.shape[0]-1, 0]/ev_correlations[ev_correlations.shape[0]-1, 0])

    V1 = np.bmat([[np.zeros((ev_correlations.shape[0]-2,2))],[[[np.cos(angle), -np.sin(angle)],[np.sin(angle),np.cos(angle)]]]] )
   
    #print(ev_correlations[:,0].T @ V1)
    #print(R.T @ sub_corr @ R)
    """


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
