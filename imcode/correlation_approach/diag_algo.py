from re import sub
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from determine_U import determine_U
from create_correlation_block import create_correlation_block
import h5py
np.set_printoptions(linewidth=np.nan, precision=2, suppress=True)

ntimes = 2
epsilon = 1.e-4


"""filename = 'Corr_t=' + str(ntimes) + '_Jx=0.3_Jy=0.3ww_g=0.0_IT'
with h5py.File(filename + '.hdf5', 'r') as f:
    key = 'corr_t='+str(ntimes)
    corr_read = f[key][:,:]"""

B = [] 
filename = '/Users/julianthoenniss/Documents/PhD/papers/correlations_paper/data/scaling_backup/paper_nb_mode=C_Jx=0.3_Jy=0.3_g=0.0mu=0.0_del_t=1.0_beta=0.0_L=200_init=2'
#filename = '/Users/julianthoenniss/Documents/PhD/papers/correlations_paper/data/scaling_backup/papermode=G_Jx=0.3_Jy=0.3_g=0.0mu=0.0_del_t=1.0_beta=0.0_L=200_init=3'
max_block_sizes = []
times = []
max_block_individual = 0
for nbr_Floquet_layers in range (1,4,2):
    with h5py.File(filename + '.hdf5', 'r') as f:
        B = f['IM_exponent'][nbr_Floquet_layers - 1,:4*nbr_Floquet_layers,:4*nbr_Floquet_layers]
    
    S = np.zeros(B.shape,dtype=np.complex_)
    for i in range (nbr_Floquet_layers):#order plus and minus next to each other
        S [B.shape[0] // 2 - (2 * i) - 2,4 * i] = 1
        S [B.shape[0] // 2 - (2 * i) - 1,4 * i + 2] = 1
        S [B.shape[0] // 2 + (2 * i) ,4 * i + 1] = 1
        S [B.shape[0] // 2 + (2 * i) + 1,4 * i + 3] = 1

    B = S @ B @ S.T

    #the following two transformation bring it into in/out- basis (not theta, zeta)
    rot = np.zeros((4 * nbr_Floquet_layers,4 * nbr_Floquet_layers))
    for i in range(0,4*nbr_Floquet_layers, 2):#go from bar, nonbar to zeta, theta
        rot[i,i] = 1./np.sqrt(2)
        rot[i,i+1] = 1./np.sqrt(2)
        rot[i+1,i] = - 1./np.sqrt(2) * np.sign(2*nbr_Floquet_layers - i-1)
        rot[i+1,i+1] = 1./np.sqrt(2) * np.sign(2*nbr_Floquet_layers - i-1)
    B = rot.T @ B @ rot
    
    #for Michael:
    U = np.zeros(B.shape)#order in the way specified in pdf for him
    for i in range (nbr_Floquet_layers):
        U[4*i, B.shape[0] //2 - (2*i) -1] = 1
        U[4*i + 1, B.shape[0] //2 + (2*i)] = 1
        U[4*i + 2, B.shape[0] //2 - (2*i) -2] = 1
        U[4*i + 3, B.shape[0] //2 + (2*i) + 1] = 1
    B = U @ B @ U.T 
    print(B.shape)
    print(B*0.5)
    corr_read = create_correlation_block(B, nbr_Floquet_layers)
        

    tol = 1.e-15
    corr_read[abs(corr_read) < tol] = 0.0
    print(np.real(corr_read))

    print(corr_read.shape)

    dim_corr = corr_read.shape[0]
    print(dim_corr)
    corr = np.zeros(corr_read.shape,dtype=np.complex_)
    U_total = np.identity(dim_corr, dtype=np.complex_)
    U_temp = np.identity(dim_corr, dtype=np.complex_)

    corr [0:dim_corr:2,0:dim_corr:2] = corr_read [0:dim_corr//2,0:dim_corr//2]
    corr [0:dim_corr:2,1:dim_corr:2] = corr_read [0:dim_corr//2,dim_corr//2:dim_corr]
    corr [1:dim_corr:2,0:dim_corr:2] = corr_read [dim_corr//2:dim_corr,0:dim_corr//2]
    corr [1:dim_corr:2,1:dim_corr:2] = corr_read [dim_corr//2:dim_corr,dim_corr//2:dim_corr]

    corr_init = corr

    tol = 1.e-15
    corr[abs(corr) < tol] = 0.0

    print(np.real(corr))


    sub_corr = []
    eigvecs = []

    for lower in range(0,dim_corr-2,2):

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
                max_block_individual = max(max_block_individual,upper-lower)


        U_temp = determine_U(sub_corr, eigvecs.T[0], lower, upper, dim_corr)
        corr = U_temp @ corr @ U_temp.T.conj()
        
        tol = 1.e-15
        diag1 = corr
        diag1[abs(diag1) < tol] = 0.0
        print('diag',diag1)

        U_total = U_temp @ U_total 

    print(np.diag(U_total @ corr_init @ U_total.T.conj()))
    max_block_sizes = np.append(max_block_sizes,max_block_individual / 2)
    times = np.append(times,nbr_Floquet_layers)


data = np.zeros((2,len(times)))
data[:,:] = [times[:],max_block_sizes[:]]
print (data)

with h5py.File(filename + "blockscaling" + ".hdf5", 'w') as f:
        dset_blocks = f.create_dataset('block_scaling', (2,len(times)),dtype=np.int_)
        hdf5_data = f['block_scaling']
        hdf5_data[:,:] = data[:,:]

"""
#read out test
with h5py.File(filename + "blockscaling" + ".hdf5", 'r') as f:
   hdf5_data = f['block_scaling']
   np.set_printoptions(linewidth=np.nan, precision=7, suppress=True)
   data_read = hdf5_data[:,:]
   print(data_read)
"""
#fig, ax = plt.subplots()
#fig.set_size_inches(12.5,10.2) 
#ax.set_xscale('log')
#ax.plot( np.arange(len(max_block_sizes)),max_block_sizes ,alpha=1, marker='o',linestyle=':')
#plt.show()