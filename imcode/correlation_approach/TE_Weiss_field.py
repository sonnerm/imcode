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
work_path = '/Users/julianthoenniss/Documents/PhD/data/DMFT_data/G0_keldysh_neqDMFT/'
fiteo1_path = '/home/thoennis/DMFT_data/ins_met_transition/'
baobab_path = '/home/users/t/thoennis/scratch/'

Weiss_data_path = '/Users/julianthoenniss/Documents/Phd/data/DMFT_data/G0_keldysh_neqDMFT/'#to read out data

#Weiss_data_path = '/home/thoennis/DMFT_data/ins_met_transition/'
ntimes = 100#len(data[:,1]) // 2 
G = np.zeros((4,ntimes,ntimes),dtype=np.complex_)

Weiss_file = 'G0_keldysh_neqDMFT'
filename = work_path + 'TE_from_' + Weiss_file + '_' + str(ntimes) #to store evaluated data

Weiss_files = ['re_G0_less_t_t.neqipt','im_G0_less_t_t.neqipt','re_G0_ret_t_t.neqipt','im_G0_ret_t_t.neqipt']

counter=0
for Weiss_file in Weiss_files:
    
    Weiss_data_file = Weiss_data_path + Weiss_file

    # Read the data.
    with open(Weiss_data_file , 'r') as fh:
        lines = fh.readlines()

    # Remove newlines, tabs, and split each string separated by spaces.
    clean = [line.strip().replace('\t', '').split() for line in lines]

    # Feed the data into a DataFrame.
    data = pd.DataFrame(clean[:], columns=clean[0]).astype(float).to_numpy()
    
    #ntimes = len(data[:,1]) // 2

    print('ntimes', ntimes)

    for i in range(0,ntimes):
        #B[i,:] = np.concatenate( (np.dot(-1,(data[range(ntimes-1 - i  , ntimes-1 ),1])) , data[range(ntimes-1 , i - 1, -1),1] ), axis=None) #data[range(ntimes-1 + i , -1+ i, -1),1]
        G[counter,i,:] = data[ntimes * i +i:ntimes * (i+1)+i,2]
    counter += 1

G_lesser = G[0,:,:] + 1.j * G[1,:,:]
G_ret = G[2,:,:] + 1.j * G[3,:,:]

#print(G_ret)

G_greater = - G_lesser.T.conj()
G_advanced = G_ret.T.conj()
G_Keldysh =  G_lesser + G_greater

L = 1/np.sqrt(2) * np.matrix('1,-1;1,1')
tau_3 =  np.matrix('1,0;0,-1')

#B=np.bmat([[G_ret,G_Keldysh], [np.zeros(G_ret.shape,dtype=np.complex_), G_advanced]])

G = np.zeros((2*ntimes,2*ntimes),dtype=np.complex_)
for t in range (ntimes):
    for tp in range (ntimes):
        
        G[2*t:2*t+2,2*tp:2*tp+2] = tau_3 @ L.T.conj() @ np.matrix([[G_ret[t,tp], G_Keldysh[t,tp]],[0,G_advanced[t,tp]]]) @ L
        
t=1
tp=3
print(np.matrix([[G_ret[t,tp], G_Keldysh[t,tp]],[0,G_advanced[t,tp]]]))
print(G[2*t:2*t+2,2*tp:2*tp+2])

t=2
tp=2
print(np.matrix([[G_ret[t,tp], G_Keldysh[t,tp]],[0,G_advanced[t,tp]]]))
print(G[2*t:2*t+2,2*tp:2*tp+2])

t=3
tp=1
print(np.matrix([[G_ret[t,tp], G_Keldysh[t,tp]],[0,G_advanced[t,tp]]]))
print(G[2*t:2*t+2,2*tp:2*tp+2])

G_inv = inv(G)#invert to obtain inverse Weiss field

#B = 0.5*np.bmat([[np.zeros(G_inv.shape), G_inv],[-G_inv.T, np.zeros(G_inv.shape)]])
B = np.zeros((4*ntimes,4*ntimes),dtype=np.complex_)

for i in range (ntimes):
    for j in range (ntimes):
        B[4*i:4*i+2,4*j+2:4*j+4] = G_inv[2*i:2*i+2, 2*j:2*j+2]
        
B = 0.5*(B - B.T)

with h5py.File(filename + ".hdf5", 'w') as f:
    dset_temp_entr = f.create_dataset('temp_entr', (1,ntimes),dtype=np.float_)
    dset_entangl_specrt = f.create_dataset('entangl_spectr', (1, ntimes,8 * ntimes),dtype=np.float_)
    dset_IM_exponent = f.create_dataset('IM_exponent', (4 * ntimes,4 * ntimes),dtype=np.complex_)#factor 4 for non-equil. data

with h5py.File(filename + '.hdf5', 'a') as f:
    Weiss_data = f['IM_exponent']
    Weiss_data[:,:] = B[:,:]

correlation_block = create_correlation_block(B, ntimes)
print('C_shape',correlation_block.shape)
time_cuts = np.arange(1, 100)

with h5py.File(filename + '.hdf5', 'a') as f:
    entr_data = f['temp_entr']
    
    for cut in time_cuts:
        print('calculating entropy at time cut:', cut)
        ent_val = float(entropy('D', correlation_block, ntimes, cut,0, filename))
        entr_data[0,cut] = ent_val
        #print('ent_val', ent_val)




with h5py.File(filename + '.hdf5', 'r') as f:
   entr_data = f['temp_entr']
   np.set_printoptions(linewidth=np.nan, precision=10, suppress=True)
   print(entr_data[:])
