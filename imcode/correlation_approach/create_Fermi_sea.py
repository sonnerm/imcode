import numpy as np
from scipy import linalg
import h5py


#define parameters:

J = 0.1
beta = 100000. 
lengths = np.arange(100,3500,100)

Jx = J
Jy = J

def compute_generators(nsites, Jx=0, Jy=0):
    # G_XY - two-site gates (XX + YY)
    G_XY_odd = np.zeros((2 * nsites, 2 * nsites),dtype=np.complex_)
    G_XY_even = np.zeros((2 * nsites, 2 * nsites),dtype=np.complex_)

    Jp = (Jx + Jy)
    Jm = (Jy - Jx)

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

    return G_XY_even, G_XY_odd



for nsites in lengths:
    G_XY_even, G_XY_odd = compute_generators(nsites, Jx, Jy)

    #XX Hamiltonian
    H_XX = 0.5 * (G_XY_even + G_XY_odd)

    #find spectrum and eigenvectors
    eigenvals, eigenvecs_dressed1 = linalg.eigh(H_XX[:nsites,:nsites])

    #matrix that diagonalizes Hamiltonian: N_E.T.conj() @ H_XX @ N_E = diag 
    N_E = np.bmat([[eigenvecs_dressed1,np.zeros((nsites,nsites))],[np.zeros((nsites,nsites)), eigenvecs_dressed1.conj()]])

    #create correlation matrix as fermi-distribution in diagonal basis
    Lambda_diag = np.zeros(H_XX.shape)
    for i in range (nsites):
        Lambda_diag[i,i] = 1./(1+np.exp(-beta * eigenvals[i]))
        Lambda_diag[i+nsites,i+nsites] = 1./(1+np.exp(beta * eigenvals[i]))

    #rotate back to real space basis
    Lambda = N_E @ Lambda_diag @ N_E.T.conj() 
    print('Correlation matrix computed for L=' + str(nsites))
    #Lambda is in block form, i.e. upper left block contains all <c c^\dag>, upper right contains all <c c>, etc..
    
    #store with a dataset per correlation matrix
    with h5py.File('FermiSea_correlations_beta={}'.format(beta) + ".hdf5", 'w') as f:
        dset_corr = f.create_dataset('Lambda_L={}'.format(nsites), (Lambda.shape),dtype=np.complex_)
        dset_corr[:,:] = Lambda[:,:]
        print('Correlation matrix stored.')
