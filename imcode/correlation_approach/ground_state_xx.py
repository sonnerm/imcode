import numpy as np
from numpy import dtype, version
from numpy.lib.type_check import imag
from scipy.linalg import expm, schur, eigvals
from scipy import linalg
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from numpy.linalg import multi_dot
import scipy.optimize
import h5py
from compute_generators import compute_generators
from matrix_diag import matrix_diag
from determine_U import determine_U

beta = 100
Jx=0.3
Jy=0.3
g=0
beta_tilde = 0
epsilon = 1.e-6
L_max = 5
L_min = 4

filename = 'H_XX_'
with h5py.File(filename + "blockscaling_eps=" + str(epsilon) + ".hdf5", 'w') as f:
    dset_blocks = f.create_dataset('block_scaling', (4,(L_max - L_min)),dtype=np.int_)
    dset_blocks = f.create_dataset('block_sizes', ((L_max - L_min),L_max),dtype=np.int_)
    dset_blocks = f.create_dataset('last_gates', ((L_max - L_min),L_max),dtype=np.int_)#1:Givens, 2:Bog
    dset_blocks = f.create_dataset('init_states', ((L_max - L_min),L_max),dtype=np.int_)

max_block_individual = 0
iter = 0
nbr_gates = 0

for nsites in range(L_min,L_max):
    G_XY_even, G_XY_odd, G_g, G_1 = compute_generators(nsites, Jx, Jy, g, beta_tilde)

    H_XX = 0.5 * (G_XY_even + G_XY_odd)

    eigenvals, eigenvecs_dressed1 = linalg.eigh(H_XX[:nsites,:nsites])

    N_E = np.bmat([[eigenvecs_dressed1,np.zeros((nsites,nsites))],[np.zeros((nsites,nsites)), eigenvecs_dressed1.conj()]])

    Lambda_diag = np.zeros(H_XX.shape)
    for i in range (nsites):
        Lambda_diag[i,i] = 1./(1+np.exp(-beta * eigenvals[i]))
        Lambda_diag[i+nsites,i+nsites] = 1./(1+np.exp(beta * eigenvals[i]))

    Lambda = N_E @ Lambda_diag @ N_E.T.conj() 

    dim_corr = Lambda.shape[0]
    corr = np.zeros(Lambda.shape,dtype=np.complex_)
    U_total = np.identity(dim_corr, dtype=np.complex_)
    U_temp = np.identity(dim_corr, dtype=np.complex_)

    corr [0:dim_corr:2,0:dim_corr:2] = Lambda [0:dim_corr//2,0:dim_corr//2]
    corr [0:dim_corr:2,1:dim_corr:2] = Lambda[0:dim_corr//2,dim_corr//2:dim_corr]
    corr [1:dim_corr:2,0:dim_corr:2] = Lambda [dim_corr//2:dim_corr,0:dim_corr//2]
    corr [1:dim_corr:2,1:dim_corr:2] = Lambda [dim_corr//2:dim_corr,dim_corr//2:dim_corr]

    #after this, Lambda is ordered as (<c^\dagger c>  & <c^\dagger c^\dagger>\\ <c c>  & <c c^\dagger>)_{ij}

    corr_init = corr

    tol = 1.e-15

    corr[abs(corr) < tol] = 0.0

    #print(np.real(corr))


    sub_corr = []
    eigvecs = []

    average_blocksize = 0
    nbr_iterations = 0
    blocksizes = []
    lastgates = []
    for lower in range(0,dim_corr-2,2):
        nbr_iterations += 1
        upper = lower + 2
        search_upper= 1
        while(search_upper ):
            upper += 2
            print('checking:',lower,upper)
            sub_corr = corr[lower:upper,lower:upper]

            eigvals, eigvecs = linalg.eigh(sub_corr)

            arg_list = np.argsort(-eigvals)
            largest_eigval = eigvals[arg_list[0]]
        
            print('largest eigval: ', largest_eigval)
            if (abs(largest_eigval - 1) > epsilon) and lower < dim_corr - 2 and upper <dim_corr :      
                print('expanding system further..')
            else:
                search_upper = 0
                print('diagonalizing subsystem..')
                max_block_individual = max(max_block_individual,(upper-lower)/2)
                average_blocksize += (upper - lower)/2
                blocksizes.append((upper - lower)/2)
                nbr_gates += ((upper - lower)/2 -1 ) * 2
    

        U_temp, last_gate_type = determine_U(sub_corr, eigvecs.T[arg_list[0]], lower, upper, dim_corr)
        lastgates.append(last_gate_type)
        corr = U_temp @ corr @ U_temp.T.conj()
        
        tol = 1.e-15
        diag1 = corr
        diag1[abs(diag1) < tol] = 0.0
        #print('diag',diag1)

        U_total = U_temp @ U_total 

    U_total[abs(U_total) < tol] = 0.0
    diag = np.round(abs(np.diag(U_total @ corr_init @ U_total.T.conj())))
    #print(diag)
    init_state = diag[0::2]
    #print(init_state)

    average_blocksize = average_blocksize / nbr_iterations #divide by number of iterations in correlation matrix
    #print(nbr_gates)
 
    with h5py.File(filename + "blockscaling_eps=" + str(epsilon) + ".hdf5", 'a') as f:
            hdf5_data = f['block_scaling']
            hdf5_data[0,iter] = nsites
            hdf5_data[1,iter] = max_block_individual 
            hdf5_data[2,iter] = average_blocksize
            hdf5_data[3,iter] = nbr_gates

            hdf5_data = f['block_sizes']
            hdf5_data[iter,:len(blocksizes)] = blocksizes[:]
            hdf5_data = f['last_gates']
            hdf5_data[iter,:len(lastgates)] = lastgates[:]
            hdf5_data = f['init_states']
            hdf5_data[iter,:nsites] = init_state[:]


    print('Iteration ', iter+1,' stored (L =', nsites,', max_blocksize=', max_block_individual,', aver_blocksize=', average_blocksize, ').')

    iter += 1