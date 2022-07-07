
import numpy as np
from scipy import linalg
import sys
from create_correlation_block import create_correlation_block
from entropy import entropy
import h5py

#filename = '/Users/julianthoenniss/Documents/PhD/data/compmode=G_o=1_Jx=0.1_Jy=0.1_g=0.0mu=0.0_del_t=1.0_beta=0.0_L=41_init=2'
filename = '/Users/julianthoenniss/Documents/PhD/data/Jx=0.1_Jy=0.1_g=0.0mu=0.0_del_t=1.0_L=200_InfTemp'
#filename = '/Users/julianthoenniss/Documents/PhD/data/compmode=C_o=1_Jx=0.1_Jy=0.1_g=0.0mu=0.0_del_t=1.0_beta=0.0_L=20_init=2'
#filename = '/Users/julianthoenniss/Documents/PhD/data/compmode=C_Jx=0.1_Jy=0.1_g=0.0mu=0.0_del_t=1.0_beta=0.0_L=200_init=3'
for iter in range(5):


    with h5py.File(filename + '.hdf5', 'r') as f:
        times_read = f['temp_entr']
        nbr_Floquet_layers  = int(times_read[iter,0])
        print('times: ', nbr_Floquet_layers )

    B = np.zeros((4*nbr_Floquet_layers,4*nbr_Floquet_layers), dtype=np.complex_)
    with h5py.File(filename + '.hdf5', 'r') as f:
        print(4*nbr_Floquet_layers,4*nbr_Floquet_layers)
        B = f['IM_exponent'][iter,:4*nbr_Floquet_layers,:4*nbr_Floquet_layers]



    dim_B = B.shape[0]
    #rotate into correct basis with input/output variables

    S = np.zeros(B.shape,dtype=np.complex_)
    for i in range (dim_B//4):#order plus and minus next to each other
        S [dim_B // 2 - (2 * i) - 2,4 * i] = 1
        S [dim_B // 2 - (2 * i) - 1,4 * i + 2] = 1
        S [dim_B // 2 + (2 * i) ,4 * i + 1] = 1
        S [dim_B // 2 + (2 * i) + 1,4 * i + 3] = 1

    #the following two transformation bring it into in/out- basis (not theta, zeta)
    rot = np.zeros(B.shape,dtype=np.complex_)
    for i in range(0,dim_B, 2):
        rot[i,i] = 1./np.sqrt(2)
        rot[i,i+1] = 1./np.sqrt(2)
        rot[i+1,i] = - 1./np.sqrt(2) * np.sign(dim_B//2 - i-1)
        rot[i+1,i+1] = 1./np.sqrt(2) * np.sign(dim_B//2 - i-1)

    B = rot.T @ S @ B @ S.T @ rot 

    #add integration measure
    #measure (this imposes the IT initial state)
    for i in range (dim_B//2-1):
        B[2 * i + 1 ,2 * i+2 ] += 1
        B[2 * i + 2 ,2 * i+1 ] -= 1

    #temporal boundary condition for measure
    B[0, dim_B - 1] += 1#sign because substituted in such a way that all kernels are the same.
    B[dim_B - 1,0] -= 1


    B_inv = linalg.inv(B)

    print(- B_inv[0,dim_B//2-1])