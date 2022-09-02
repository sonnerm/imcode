
import numpy as np
from scipy import linalg
import sys

import h5py

# seed random number generator
def zero_to_nan(values):
    """Replace every 0 with 'nan' and return a copy."""
    return [float('nan') if x==0 else x for x in values]


np.set_printoptions(threshold=sys.maxsize, precision=3)

filename = '/Users/julianthoenniss/Documents/PhD/data/Millis_interleaved_timestep=0.1_hyb=0.05_T=50-200_Delta=1'

conv = 'M'

if conv == 'J':
    #filename += '_my_conv' 
    print('using Js convention')
elif conv == 'M':
    filename += '_Michaels_conv' 
    print('using Ms convention')


max_time = 200
interval = 1
Gamma = 0.05
delta_t = 0.1
t = 0*1.1 * delta_t # hopping between spin species, factor 2 to match Michael's spin convention

nbr_iterations = 4

with h5py.File(filename+'_square_norm' + ".hdf5", 'w') as f:
    dset_square_norm = f.create_dataset('square_norm', (nbr_iterations,), dtype=np.float_)
    dset_norm_time = f.create_dataset('iterations', (nbr_iterations,), dtype=int)

square_norm = np.zeros(nbr_iterations,dtype=np.complex_)
times = np.zeros(nbr_iterations,dtype=np.int_)

for iter in range(0,nbr_iterations):

    iter_readout =  iter
    with h5py.File(filename + '.hdf5', 'r') as f:
        #times_read = f['temp_entr']
        times_read = f['times']
        nbr_Floquet_layers= int(times_read[iter_readout])
        print('times: ', nbr_Floquet_layers)

    times[iter] = nbr_Floquet_layers


    B = np.zeros((4*nbr_Floquet_layers, 4*nbr_Floquet_layers),
                 dtype=np.complex_)
    with h5py.File(filename + '.hdf5', 'r') as f:
        print(4*nbr_Floquet_layers, 4*nbr_Floquet_layers)
        B = f['IM_exponent'][iter_readout, :4*nbr_Floquet_layers, :4*nbr_Floquet_layers]

    action = np.zeros((2*B.shape[0],2*B.shape[0]),dtype=np.complex_)

    action[:B.shape[0],:B.shape[0]] = B.conj().T[:,:]
    action[B.shape[0]:,B.shape[0]:] = B[:,:]
    #measure
    action[:B.shape[0],B.shape[0]:] = +np.identity(B.shape[0])
    action[B.shape[0]:,:B.shape[0]] = -np.identity(B.shape[0])

    print("computing norm and storing..")
    with h5py.File(filename+'_square_norm' + ".hdf5", 'a') as f:
        norm_data = f['square_norm']
        norm_data[iter] = np.real(np.sqrt(linalg.det(action)))
        print('norm at Floquet_layer ' , str(nbr_Floquet_layers), ': ', str(norm_data[iter]))
        norm_times = f['iterations']
        norm_times[iter] = nbr_Floquet_layers
        print("stored data for iteration: ", str(nbr_Floquet_layers))