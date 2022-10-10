
from bisect import bisect_left
from email import iterators
import numpy as np
from scipy import linalg
import sys
from create_correlation_block import create_correlation_block
import h5py
import matplotlib.pyplot as plt
# seed the pseudorandom number generator
from random import seed
from random import random
from pfapack import pfaffian as pf
import pandas as pd
# seed random number generator
def zero_to_nan(values):
    """Replace every 0 with 'nan' and return a copy."""
    return [float('nan') if x==0 else x for x in values]


np.set_printoptions(threshold=sys.maxsize, precision=6)
filename_right = '/Users/julianthoenniss/Documents/PhD/data/corr_mich/2809/current/finite_bias/compmode=C_o=1_Jx=1.0_Jy=1.0_g=0.0mu=0.2_del_t=0.1_beta=200.0_L=60_init=4_coupling=0.5'
filename_left = '/Users/julianthoenniss/Documents/PhD/data/corr_mich/2809/current/finite_bias/compmode=C_o=2_Jx=1.0_Jy=1.0_g=0.0mu=-0.2_del_t=0.1_beta=200.0_L=60_init=4_coupling=0.5'

potentials = [0.0,0.005,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.055,0.06,0.07]

for potential in potentials:
    mu = potential /2
    filename_right = '/Users/julianthoenniss/Documents/PhD/data/compmode=C_o=1_Jx=1.0_Jy=1.0_g=0.0mu=' +str(mu) +'_del_t=0.025_beta=10000.0_L=300_init=4_coupling=0.3162'
    filename_left = '/Users/julianthoenniss/Documents/PhD/data/compmode=C_o=2_Jx=1.0_Jy=1.0_g=0.0mu=-' +str(mu) +'_del_t=0.025_beta=10000.0_L=300_init=4_coupling=0.3162'
    filename_output = '/Users/julianthoenniss/Documents/PhD/data/benchmark_GM_current_' + str(potential) + '_0.025'

    conv = 'J'

    if conv == 'J':
        #filename += '_my_conv' 
        print('using Js convention')
    elif conv == 'M':
        filename += '_Michaels_conv' 
        print('using Ms convention')


    max_time =300
    interval = 1
    delta_t = 0.025
    mu = 0.

    with h5py.File(filename_output+'_spinfulpropag' + ".hdf5", 'w') as f:
        dset_I_right = f.create_dataset('I_right', (max_time,), dtype=np.complex_)
        dset_I_left = f.create_dataset('I_left', (max_time,), dtype=np.complex_)
        dset_I_left = f.create_dataset('n_tot', (max_time,), dtype=np.complex_)
        dset_propag_exact = f.create_dataset('propag_times', (max_time,), dtype=np.float_)


    times = np.zeros(max_time,dtype=np.int_)

    for iter in range(0,1,interval):

        iter_readout = iter
        with h5py.File(filename_left + '.hdf5', 'r') as f:

            #read out time from chain:
            times_read = f['temp_entr']
            nbr_Floquet_layers  = int(times_read[iter,0])

            #read out time from specdens
            #times_read = f['times']
            #nbr_Floquet_layers= int(times_read[iter_readout])#300
            #nbr_Floquet_layers = iter_readout + 1
            print('times: ', nbr_Floquet_layers)
        max_time = nbr_Floquet_layers
        times[iter] = nbr_Floquet_layers
        intermediate_time = nbr_Floquet_layers
        total_time = nbr_Floquet_layers
    
        B_left= np.zeros((4*nbr_Floquet_layers, 4*nbr_Floquet_layers),
                    dtype=np.complex_)

        B_right= np.zeros((4*nbr_Floquet_layers, 4*nbr_Floquet_layers),
                    dtype=np.complex_)

        with h5py.File(filename_left + '.hdf5', 'r') as f:
            print(4*nbr_Floquet_layers, 4*nbr_Floquet_layers)
            B_left = f['IM_exponent'][iter_readout, :4*nbr_Floquet_layers, :4*nbr_Floquet_layers]
        with h5py.File(filename_right + '.hdf5', 'r') as f:
            print(4*nbr_Floquet_layers, 4*nbr_Floquet_layers)
            B_right = f['IM_exponent'][iter_readout, :4*nbr_Floquet_layers, :4*nbr_Floquet_layers]
        
    
        dim_B = B_left.shape[0]

        print('dim_b',dim_B)
        # rotate into Grassmann basis with input/output variables

        if conv == 'M':
            U = np.zeros(B_left.shape)#order in the way specified in pdf for him (forward,backward,forward,backward,...)
            for i in range (nbr_Floquet_layers):
                U[4*i, B_left.shape[0] //2 - (2*i) -1] = 1
                U[4*i + 1, B_left.shape[0] //2 + (2*i)] = 1
                U[4*i + 2, B_left.shape[0] //2 - (2*i) -2] = 1
                U[4*i + 3, B_left.shape[0] //2 + (2*i) + 1] = 1
            B_left = U.T @ B_left @ U
            B_right = U.T @ B_right @ U
            print('Rotated from M basis to Grassmann basis')

        else:
        #rotate into correct basis with input/output variables
            S = np.zeros(B_left.shape,dtype=np.complex_)
            for i in range (dim_B//4):#order plus and minus next to each other
                S [dim_B // 2 - (2 * i) - 2,4 * i] = 1
                S [dim_B // 2 - (2 * i) - 1,4 * i + 2] = 1
                S [dim_B // 2 + (2 * i) ,4 * i + 1] = 1
                S [dim_B // 2 + (2 * i) + 1,4 * i + 3] = 1

            #the following two transformation bring it into in/out- basis (not theta, zeta)
            rot = np.zeros(B_left.shape,dtype=np.complex_)
            for i in range(0,dim_B, 2):
                rot[i,i] = 1./np.sqrt(2)
                rot[i,i+1] = 1./np.sqrt(2)
                rot[i+1,i] = - 1./np.sqrt(2) * np.sign(dim_B//2 - i-1)
                rot[i+1,i+1] = 1./np.sqrt(2) * np.sign(dim_B//2 - i-1)
            B_left = rot.T @ S @ B_left @ S.T @ rot 
            B_right = rot.T @ S @ B_right @ S.T @ rot 
            print('Rotated from J basis to Grassmann basis')


        #here, the matrix B is in the Grassmann convention
    
        exponent = np.zeros((4*dim_B, 4*dim_B), dtype=np.complex_)#this exponent will contain the exponents of both spin species as well as the impurity dynamics
        # Influence matices for both spin species
        #spin down
        exponent[:dim_B, :dim_B] = B_left[:, :]
        #spin up
        exponent[3*dim_B:4*dim_B, 3*dim_B:4*dim_B] = B_right[:, :]

    
        # integration measure
        # Left
        exponent[2*dim_B:3*dim_B-1,1:dim_B] += -np.identity(dim_B-1)
        exponent[3*dim_B-1,0] += -1
        exponent[1:dim_B,2*dim_B:3*dim_B-1] += +np.identity(dim_B-1)
        exponent[0,3*dim_B-1] += +1

        # Right
        exponent[3*dim_B:,dim_B:2*dim_B] += +np.identity(dim_B)
        exponent[dim_B:2*dim_B,3*dim_B:] += -np.identity(dim_B)
        
        # (thermal with temperature beta)
        beta = 400
        #initial state only connected to right IM
        exponent[dim_B + dim_B//2 - 1, dim_B + dim_B//2 ] += np.exp(beta) *(-1.)
        #Transpose (antisymm)
        exponent[dim_B + dim_B//2, dim_B + dim_B//2 - 1] -= np.exp(beta) *(-1.)
        


        
        # temporal boundary condition for measure (antiperiodic)
        # sign because substituted in such a way that all kernels are the same.
        #temporal boudary condition only connected to left IM
        exponent[3 * dim_B - 2, 3 * dim_B - 1] += -1 *(-1.)
        #Transpose (antisymm)
        exponent[3 * dim_B - 1, 3 * dim_B - 2] -= - 1 *(-1.)

        #impurity
        #hopping between spin species (gate is easily found by taking kernel of xy gate at isotropic parameters):

        for i in range(dim_B//4-1):
            print(i, nbr_Floquet_layers,2 * dim_B + dim_B//2 - 2 - 2*i - 2*dim_B)
            #minus signs *(-1.) behind lines come from overlap form
            #forward
            exponent[2 * dim_B + dim_B//2 - 2 - 2*i, dim_B + dim_B//2 - 2 - 2*i] += np.exp(1.j * mu) *(-1.)
            exponent[dim_B + dim_B//2 - 3 - 2*i, 2 * dim_B + dim_B//2 - 3 - 2*i] += np.exp(1.j * mu) *(-1.)
            # forward Transpose (antisymm)
            exponent[dim_B + dim_B//2 - 2 - 2*i, 2 * dim_B + dim_B//2 - 2 - 2*i] += -np.exp(1.j * mu) *(-1.)
            exponent[2 * dim_B + dim_B//2 - 3 - 2*i, dim_B + dim_B//2 - 3 - 2*i] += -np.exp(1.j * mu) *(-1.)

            #backward
            exponent[2 * dim_B + dim_B//2 - 1 + 2*i, dim_B + dim_B//2 +1 + 2*i] += -np.exp(-1.j * mu) *(-1.)
            exponent[dim_B + dim_B//2 +2 + 2*i, 2 * dim_B + dim_B//2 + 2*i ] += -np.exp(-1.j * mu) *(-1.)
            # backward Transpose (antisymm)
            exponent[dim_B + dim_B//2 +1 + 2*i, 2 * dim_B + dim_B//2 -1 + 2*i] += np.exp(-1.j * mu) *(-1.)
            exponent[2 * dim_B + dim_B//2 + 2*i , dim_B + dim_B//2 +2 + 2*i] += np.exp(-1.j * mu) *(-1.)

        #last gate forward 
        exponent[2 * dim_B, dim_B ] += np.exp(1.j * mu) *(-1.)
        # forward Transpose (antisymm)
        exponent[dim_B, 2*dim_B ] += - np.exp(1.j * mu) *(-1.)
        #backward
        exponent[2 * dim_B -1, 3*dim_B - 3 ] += np.exp(1.j * mu) *(-1.)
        # backward Transpose (antisymm)
        exponent[3*dim_B - 3,2 * dim_B -1 ] += -np.exp(1.j * mu) *(-1.)

        exponent_inv = linalg.inv(exponent)

        with h5py.File(filename_output+'_spinfulpropag' + ".hdf5", 'a') as f:
            I_right = f['I_right']
            I_left = f['I_left']
            n_tot = f['n_tot']
            tau0=0
            for tau in range (tau0,nbr_Floquet_layers):
                n_before = - pf.pfaffian(np.array(pd.DataFrame(exponent_inv.T).iloc[[1*dim_B + dim_B//2 -1 - 2*tau,2*dim_B + dim_B//2 -1 - 2*tau], [1*dim_B + dim_B//2 -1 - 2*tau,2*dim_B + dim_B//2 -1 - 2*tau]]))
                n_after_even = - pf.pfaffian(np.array(pd.DataFrame(exponent_inv.T).iloc[[2*dim_B + dim_B//2 -2 - 2*tau,1*dim_B + dim_B//2 -2 - 2*tau], [2*dim_B + dim_B//2 -2 - 2*tau,1*dim_B + dim_B//2 -2 - 2*tau]]))
                n_after_odd = - pf.pfaffian(np.array(pd.DataFrame(exponent_inv.T).iloc[[1*dim_B + dim_B//2 -3 - 2*tau,2*dim_B + dim_B//2 -3 - 2*tau], [1*dim_B + dim_B//2 -3 - 2*tau,2*dim_B + dim_B//2 -3 - 2*tau]]))
                
                I_right[tau] = (n_after_even - n_before)/delta_t 
                I_left[tau] = (n_after_odd - n_after_even)/delta_t
                indices = [2*dim_B + dim_B//2 -2 - 2*tau,1*dim_B + dim_B//2 -2 - 2*tau]
                n_tot[tau] = n_before
                print(n_tot[tau])
                print(I_left[tau])
                times_data = f['propag_times']
                times_data[tau - tau0] = (tau -tau0)*delta_t #nbr_Floquet_layers



