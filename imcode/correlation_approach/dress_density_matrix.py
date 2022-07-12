import numpy as np
from numpy.linalg import eig, matrix_power, det
from scipy.linalg import expm,logm
from scipy import linalg
import matplotlib.pyplot as plt
from reorder_eigenvecs import reorder_eigenvecs
from add_cmplx_random_antisym import add_cmplx_random_antisym
from scipy.linalg import sqrtm
import h5py


def dress_density_matrix(rho_0_exponent, F_E_prime, F_E_prime_dagger,M,M_E, eigenvalues_G_eff,  eigenvalues_G_eff_E, beta_tilde, nbr_Floquet_layers,init_state, G_XY_even,G_XY_odd, order, beta,mu):
    nsites = int (len(rho_0_exponent[0]) / 2)

    rho_0_single_body = 0.25*expm(- rho_0_exponent) @ expm(- rho_0_exponent)#rho_exponent = 0 corresponds to infinite temperature (factor 0.25 so it gives correct normalization 0.5 from partition function when pulled out of the square root below)

    if init_state == 2:
        rho_0_single_body = M_E @ np.bmat([[np.identity(nsites),np.zeros((nsites,nsites))],[np.zeros((nsites,nsites)),np.zeros((nsites,nsites))] ]) @ M_E.T.conj() 
    elif init_state == 4:
        #cutoff = int((2*nsites) //3)
        #negative eigenvalues first..
        diag = np.diag(np.exp(-  beta * (eigenvalues_G_eff_E[:] - mu)) / (1 + np.exp(- beta * (eigenvalues_G_eff_E[:] - mu))))
        diag = diag @ diag #since later, we have to take the square root
        #distr_func1 = np.diag(np.concatenate((np.full((1, cutoff), 1), np.full((1, nsites - cutoff), .25)), axis=None))
        #distr_func2 = np.diag(np.concatenate((np.full((1, cutoff), 0), np.full((1, nsites - cutoff), .25)), axis=None))
        #rho_0_single_body = M_E @ np.bmat([[distr_func1,np.zeros((nsites,nsites))],[np.zeros((nsites,nsites)),  distr_func2] ]) @ M_E.T.conj() 
        rho_0_single_body = M_E @ diag @ M_E.T.conj()
    
    # the following line assumes that rho_0_exponent is given in real space basis..
    dress_initial = np.identity(rho_0_single_body.shape[0])
    if order == 2:#the dressing of the initial state must be changed when first, the even layer is applied.
        #layer of even gates on left side
        dress_initial = expm(1.j* G_XY_even)
    rho_dressed =  sqrtm(matrix_power(F_E_prime,nbr_Floquet_layers)  @ dress_initial @rho_0_single_body  @ dress_initial.T.conj() @  matrix_power( F_E_prime_dagger,nbr_Floquet_layers) )

    #rho_dressed =  sqrtm( rho_0_single_body )#this is the quantity that Michael needs for exact calculation


    eigenvals_dressed, eigenvecs_dressed = linalg.eigh(rho_dressed)

    argsort = np.argsort(-eigenvals_dressed)
    
    cum = 0
    for i in range(nsites):
        cum += abs(eigenvals_dressed[i] - eigenvals_dressed[i + nsites])

    if cum > 1e-6:#if half of the spectrum is zero (as for Bog. eigenstate at critical point in xy model, the eigenvectors of zero eigenvalues are messed up. Therefore, construct rotation matric only from other eigenvectors with nonzero eigenvalues (possible by symmetrey). Order in such a way that die diagonalized matrix has first nsitmes eigenvalues eqaul to zero and others the remaining ones)
    #By knowledge of the structure of N, i.e. (N = [[A, B^*],[B, A^*]]), we can construct the right part of the matrix from the left part of the matrix "eigenvecs_dressed", such that all phase factors and the order of eigenvectors are as desired.
        N_t =  np.bmat([[eigenvecs_dressed[nsites:2*nsites,argsort[:nsites]].conj(), eigenvecs_dressed[0:nsites,argsort[:nsites]]],[eigenvecs_dressed[0:nsites,argsort[:nsites]].conj(),eigenvecs_dressed[nsites:2*nsites,argsort[:nsites]]]]) 
        eigenvals_dressed[:] = np.concatenate((eigenvals_dressed[argsort[2*nsites-1:nsites-1:-1]],eigenvals_dressed[argsort[:nsites]]))
    else: 
        N_t = np.identity(eigenvecs_dressed.shape[0])

    #print ('N_t\n',N_t)
    #N_t = eigenvecs_dressed
    #diag_check = N_t.T.conj() @ rho_dressed @ N_t

    
    #eigenvals_dressed = np.real(np.diag(diag_check))
    #print('Dressed density matrix diagonalized \n', np.real(diag_check),'\n',np.imag(diag_check))
    #print('Dressed density matrix diagonalized2 \n', eigenvecs_dressed.T.conj() @ rho_dressed @ eigenvecs_dressed)
    #print('Eigenvalues of exponent:', eigenvals_dressed)

    np.set_printoptions(linewidth=np.nan, precision=6, suppress=True)
    n_expect = np.zeros((2 * nsites))#fermi-Dirac distribution for modes (basis in which dressed density matrix is diagonal). 
    #Note that the true expectation value has an additional factor Z_over_Z0. This factor, however, does not enter the exponent of the IM and is therefore not included here.
    for k in range (nsites):
        #n_expect[k] = np.exp(+eval)  / (2 * np.cosh(eval) ) # for < c^dagger c >
       
        n_expect[k] = eigenvals_dressed[k] / (eigenvals_dressed[k] + eigenvals_dressed[k+nsites]) ## for < c^dagger c >
        n_expect[k + nsites] =  eigenvals_dressed[k+nsites]  / (eigenvals_dressed[k]+ eigenvals_dressed[k+nsites])# # for < c c^dagger >
        #print(eval)
    #print('nexpext',n_expect)
    """
    corr_real_space_diag = np.diag(n_expect)
    print(corr_real_space_diag)
    corr_real_space = N_t @ corr_real_space_diag @ N_t.T.conj()
    print(corr_real_space)
    filename_correlations =  'Jx=0.3_Jy=0.3_g=0_L=600_FermiSea_correlations'
    with h5py.File(filename_correlations + ".hdf5", 'a') as f:
        dset_corr = f.create_dataset('corr_realspace=', (corr_real_space.shape[0],corr_real_space.shape[1]),dtype=np.complex_)
        dset_corr[:,:] = corr_real_space[:,:]
    print('Real space correlations stored for Michael.')
    """

    #Z_t =0
    #for i in range (nsites):
    #    Z_t += (eigenvals_dressed[k] + eigenvals_dressed[k+nsites])
    #Z_t = 0np.trace(rho_dressed)
    #print(n_expect)
    return n_expect, N_t, rho_dressed#/Z_t
