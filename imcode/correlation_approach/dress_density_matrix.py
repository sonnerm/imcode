import numpy as np
from numpy.linalg import eig, matrix_power, det
from scipy.linalg import expm,logm
from scipy import linalg
import matplotlib.pyplot as plt
from reorder_eigenvecs import reorder_eigenvecs
from add_cmplx_random_antisym import add_cmplx_random_antisym
from scipy.linalg import sqrtm, block_diag
import h5py

np.set_printoptions(linewidth=np.nan, precision=2, suppress=True)

def dress_density_matrix(rho_0_exponent, F_E_prime, F_E_prime_dagger,M,M_E, eigenvalues_G_eff,  eigenvalues_G_eff_E, beta_tilde, nbr_Floquet_layers,init_state, G_XY_even,G_XY_odd, order, beta,mu,G_eff_E,Jx,Jy):
    nsites = int (len(rho_0_exponent[0]) / 2)

    rho_0_single_body_squared = np.zeros((2*nsites,2*nsites),dtype=np.clongdouble)

    if init_state == 2:#zero temperature
        #(negative eigenvalues come first)
        rho_0_single_body_squared = M_E @ np.bmat([[np.identity(nsites),np.zeros((nsites,nsites))],[np.zeros((nsites,nsites)),np.zeros((nsites,nsites))] ]) @ M_E.T.conj() 
    
    #infinite temperature
    elif init_state ==3:
        #just normalized identity matrix:
        rho_0_single_body_squared =.25*np.identity(2*nsites)#rho_exponent = 0 corresponds to infinite temperature (factor 0.25 so it gives correct normalization 0.5 from partition function when pulled out of the square root below)
    #print('G_eff',G_eff_E)
    #finite temperature/chemical potential
    if init_state == 4:
        #rho_0_single_body_squared = expm(-beta * (G_eff_E )) 
        #rho_0_single_body_squared *= 1./np.trace(rho_0_single_body_squared)
          # the following line assumes that rho_0_exponent is given in real space basis..
        rho_0_single_body_squared = np.zeros((2*nsites,2*nsites),dtype=np.complex_)
        for k in range (nsites):
            rho_0_single_body_squared[k,k] = 1/(1 + np.exp(beta * (eigenvalues_G_eff_E[k] - eigenvalues_G_eff_E[k+nsites] - mu ))) #factor of 2 from squaring is already included in G_eff_E
            rho_0_single_body_squared[k+nsites,k+nsites] = 1/(1 + np.exp(-beta * (eigenvalues_G_eff_E[k] - eigenvalues_G_eff_E[k+nsites] - mu )))
        rho_0_single_body_squared =  M_E @ rho_0_single_body_squared @ rho_0_single_body_squared @ M_E.T.conj()

    dress_initial = np.identity(rho_0_single_body_squared.shape[0])
    if order == 2:#the dressing of the initial state must be changed when first, the even layer is applied.
        #layer of even gates on left side
        dress_initial = expm(1.j* G_XY_even)


    rho_dressed_squared = matrix_power(F_E_prime,nbr_Floquet_layers)  @ dress_initial @ rho_0_single_body_squared  @ dress_initial.T.conj() @  matrix_power( F_E_prime_dagger,nbr_Floquet_layers)
    rho_dressed =  sqrtm(rho_dressed_squared )
    rho_dressed_log =  0.5 * logm(rho_dressed_squared )
    #print('rho_dressed',rho_dressed)
    #rho_dressed =  sqrtm( rho_0_single_body )#this is the quantity that Michael needs for exact calculation
    

    if False:#Jx == Jy:
        
        eigvals1, eigvecs1 = linalg.eigh(rho_dressed[:nsites,:nsites])
        eigvals2, eigvecs2 = linalg.eigh(rho_dressed[nsites:,nsites:])

        N_t = np.bmat([[eigvecs1,np.zeros((nsites,nsites))],[np.zeros((nsites,nsites)), eigvecs2]])
        eigenvals_dressed = np.concatenate((eigvals1,eigvals2))
        print('asdfadf',eigenvals_dressed)
    else:
        eigenvals_dressed, eigenvecs_dressed = linalg.eigh(rho_dressed)
        argsort = np.argsort(-eigenvals_dressed)
        #print(eigenvals_dressed[argsort[:]])
        cum = 0
        for i in range(nsites):
            cum += abs(eigenvals_dressed[i] - eigenvals_dressed[i + nsites])
        if cum > 1e-6:#if half of the spectrum is zero (as for Bog. eigenstate at critical point in xy model, the eigenvectors of zero eigenvalues are messed up. Therefore, construct rotation matric only from other eigenvectors with nonzero eigenvalues (possible by symmetrey). Order in such a way that die diagonalized matrix has first nsitmes eigenvalues eqaul to zero and others the remaining ones)
        #By knowledge of the structure of N, i.e. (N = [[A, B^*],[B, A^*]]), we can construct the right part of the matrix from the left part of the matrix "eigenvecs_dressed", such that all phase factors and the order of eigenvectors are as desired.
            N_t =  np.bmat([[eigenvecs_dressed[nsites:2*nsites,argsort[:nsites]].conj(), eigenvecs_dressed[0:nsites,argsort[:nsites]]],[eigenvecs_dressed[0:nsites,argsort[:nsites]].conj(),eigenvecs_dressed[nsites:2*nsites,argsort[:nsites]]]]) 
            eigenvals_dressed[:] = np.concatenate((eigenvals_dressed[argsort[2*nsites-1:nsites-1:-1]],eigenvals_dressed[argsort[:nsites]]))
        else: 
            N_t = np.identity(eigenvecs_dressed.shape[0])


    #Note that the true expectation value has an additional factor Z_over_Z0. This factor, however, does not enter the exponent of the IM and is therefore not included here.
    norm = 1
    #print(eigenvals_dressed)
    #print(eigenvalues_G_eff_E)
    """if init_state == 4:#take care of Z_0 which has not been included anywhere, here
        #Z_t / Z_0
        Z_t =1
        Z_0 =1
        for k in range (nsites):
            #print(k,eigenvals_dressed[k],eigenvals_dressed[k+nsites])
            Z_t *= (np.exp(eigenvals_dressed[k]) + np.exp(eigenvals_dressed[k+nsites]))
            Z_0 *= (np.exp(-beta*(eigenvalues_G_eff_E[k])) + np.exp(-beta*(eigenvalues_G_eff_E[k+nsites])))
        norm *= 1#Z_t #/ Z_0
        """
    #print ('N_t\n',N_t)
    diag_check = N_t.T.conj() @ rho_dressed @ N_t
    #print('diag',diag_check)
    #print(eigenvals_dressed)

    
    np.set_printoptions(linewidth=np.nan, precision=6, suppress=True)
    n_expect = np.zeros((2 * nsites))#fermi-Dirac distribution for modes (basis in which dressed density matrix is diagonal). 
  
    if False:#init_state ==4:
        for k in range (nsites):
            n_expect[k] = 1. / (1 + np.exp((eigenvals_dressed[nsites + k] - eigenvals_dressed[k] )))#*Z_t  # for < c^dagger c >
            n_expect[k + nsites] =  1. / (1 + np.exp((- eigenvals_dressed[nsites + k] + eigenvals_dressed[k]))) #*Z_t # for < c c^dagger >
        print('Z_t',Z_t)

    else:
        for k in range (nsites):
            n_expect[k] = eigenvals_dressed[k] / (eigenvals_dressed[k] + eigenvals_dressed[k+nsites]) # for < c^dagger c >
            n_expect[k + nsites] =  eigenvals_dressed[k+nsites] / (eigenvals_dressed[k]+ eigenvals_dressed[k+nsites])  # for < c c^dagger >
        
    #print(n_expect)
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
    return n_expect, N_t, rho_dressed
