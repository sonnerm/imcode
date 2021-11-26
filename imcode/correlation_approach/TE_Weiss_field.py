from Lohschmidt import Lohschmidt
from evolution_matrix import evolution_matrix
from compute_generators import compute_generators
import numpy as np
import h5py
from create_correlation_block import create_correlation_block
from entropy import entropy
from IM_exponent import IM_exponent
from add_cmplx_random_antisym import add_cmplx_random_antisym
from plot_entropy import plot_entropy
from matrix_diag import matrix_diag
from dress_density_matrix import dress_density_matrix
from compute_generators import compute_generators
from Lohschmidt import Lohschmidt
from ising_gamma import ising_gamma
import sys
from numpy.linalg import inv
import pandas as pd
from gm_integral import gm_integral
import os
import math


def guess(text):
    for t in text.split(','):
        for typ in (int, float, str):
            try:
                yield typ(t)
                break
            except ValueError as e:
                pass

np.set_printoptions(linewidth=np.nan, precision=5, suppress=True)

# define fixed parameters:
# step sizes for total times t







#set location for data storage
mac_path = '/Users/julianthoenniss/Documents/Studium/PhD/data/correlation_approach/'
work_path = '/Users/julianthoenniss/Documents/PhD/data/'
fiteo1_path = '/home/thoennis/DMFT_data/'
baobab_path = '/home/users/t/thoennis/scratch/'

Weiss_data_path = '/Users/julianthoenniss/Documents/PhD/data/DMFT_data/weiss_field_test/'
Weiss_file = 'Weiss_tau.dat'

Weiss_data_file = Weiss_data_path + Weiss_file

filename = work_path + 'TE_from_' + Weiss_file


#Weiss_data_file = '/home/thoennis/DMFT_data/Weiss_tt.dat'


# Read the data.
with open(Weiss_data_file, 'r') as fh:
    lines = fh.readlines()

# Remove newlines, tabs, and split each string separated by spaces.
clean = [line.strip().replace('\t', '').split() for line in lines]

# Feed the data into a DataFrame.
data = pd.DataFrame(clean[:], columns=clean[0]).to_numpy()

ntimes = len(data[:,1]) // 2 
B = np.zeros((ntimes,ntimes))

for i in range(0,ntimes):
    B[i,:] = data[range(ntimes-1 + i , -1+ i, -1),1]

B = inv(B)#invert to obtain Weiss field
with h5py.File(filename + ".hdf5", 'w') as f:
    dset_temp_entr = f.create_dataset('temp_entr', (1,ntimes),dtype=np.float_)
    dset_entangl_specrt = f.create_dataset('entangl_spectr', (1,2 * ntimes,2 * ntimes),dtype=np.float_)
    dset_IM_exponent = f.create_dataset('IM_exponent', (ntimes,ntimes),dtype=np.float_)

correlation_block = create_correlation_block(B, ntimes)
time_cuts = np.arange(1, 80)

with h5py.File(filename + '.hdf5', 'a') as f:
    entr_data = f['temp_entr']
    Weiss_data = f['IM_exponent']

    Weiss_data[:,:] = B[:,:]
    
    for cut in time_cuts:
        print('calculating entropy at time cut:', cut)
        ent_val = float(entropy(correlation_block, ntimes, cut,0, filename))
        entr_data[0,cut] = ent_val
        print('ent_val', ent_val)




with h5py.File(filename + '.hdf5', 'r') as f:
   entr_data = f['temp_entr']
   np.set_printoptions(linewidth=np.nan, precision=10, suppress=True)
   print(entr_data[:])
