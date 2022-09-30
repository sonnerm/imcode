
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
# seed random number generator
def zero_to_nan(values):
    """Replace every 0 with 'nan' and return a copy."""
    return [float('nan') if x==0 else x for x in values]

filename = '/Users/julianthoenniss/Documents/PhD/data/B_imag'


Gamma = 1.
delta_t = 0.1
t = 0.# * delta_t#-3*0.5*delta_t#-100#4 * delta_t # hopping between spin species, factor 2 to match Michael's spin convention

dim_B = 14

"""B = np.zeros((dim_B,dim_B),dtype=np.complex_)
for i in range (dim_B):
    for j in range (dim_B):
        B[i,j] = np.tan(3.2*i+2.2*j)"""

B = np.random.rand(dim_B,dim_B) + 1.j * np.random.rand(dim_B,dim_B) 
B -= B.T#this is twice the exponent matrix
print(B)
exponent = np.zeros((4*dim_B, 4*dim_B), dtype=np.complex_)#this exponent will contain the exponents of both spin species as well as the impurity dynamics
# Influence matices for both spin species
#spin down
exponent[dim_B:2*dim_B, dim_B:2*dim_B] = B[:, :]
#spin up
exponent[2*dim_B:3*dim_B, 2*dim_B:3*dim_B] = B[:, :]

# integration measure
# spin up
exponent[2*dim_B:3*dim_B-1,1:dim_B] += -np.identity(dim_B-1)
exponent[3*dim_B-1,0] += -1
exponent[1:dim_B,2*dim_B:3*dim_B-1] += +np.identity(dim_B-1)
exponent[0,3*dim_B-1] += +1
# spin down
exponent[3*dim_B+1:4*dim_B,dim_B:2*dim_B-1] += -np.identity(dim_B-1)
exponent[3*dim_B,2*dim_B-1] += -1
exponent[dim_B:2*dim_B-1,3*dim_B+1:4*dim_B] += +np.identity(dim_B-1)
exponent[2*dim_B-1,3*dim_B] += +1



exponent_check = exponent.copy()
#impurity
#hopping between spin species (gate is easily found by taking kernel of xy gate at isotropic parameters):
seed(10)
for i in range(dim_B//2 -1):
    
    mu_up =0.52#4*delta_t#0.3*delta_t# 0.3*np.sin(2.2 * i)
    mu_down =0.52#4*delta_t#0.3*delta_t# 0.18*np.sin(1.82 * i)
  
    T=1+np.tan(t/2)**2
    # forward 
    # (matrix elements between up -> down), last factors of (-1) are sign changes to test overlap form
    exponent_check[dim_B - 2 - 2*i, 4*dim_B - 1 - 2*i] += -1. * np.tan(t/2) *2/T *np.exp(-1. * mu_up) 
    exponent_check[dim_B - 1 - 2*i, 4*dim_B - 2 - 2*i] -= -1. * np.tan(t/2)*2/T *np.exp(-1. * mu_down) 
    #(matrix elements between up -> up)
    exponent_check[dim_B - 2 - 2*i, dim_B - 1 - 2*i] += 1 *np.cos(t) *np.exp(-1 * mu_up) *(-1.) 
    #(matrix elements between down -> down)
    exponent_check[4*dim_B - 2 - 2*i, 4*dim_B - 1 - 2*i] += 1 *np.cos(t) *np.exp(-1. * mu_down) *(-1.)

    # forward Transpose (antisymm)
    exponent_check[4*dim_B - 1 - 2*i, dim_B - 2 - 2*i] += 1 * np.tan(t/2)*2/T *np.exp(-1 * mu_up) 
    exponent_check[4*dim_B - 2 - 2*i, dim_B - 1 - 2*i] -= 1. * np.tan(t/2)*2/T *np.exp(-1. * mu_down)
    exponent_check[dim_B - 1 - 2*i,dim_B - 2 - 2*i] += -1 *np.cos(t) *np.exp(-1. * mu_up) *(-1.)
    exponent_check[4*dim_B - 1 - 2*i, 4*dim_B - 2 - 2*i] += -1 *np.cos(t) *np.exp(-1. * mu_down) *(-1.)

#last application contains antiperiodic bc.:
exponent_check[0, 3*dim_B +1] += -1. * np.tan(t/2) *2/T *np.exp(-1. * mu_up) *(-1.) 
exponent_check[1, 3*dim_B ] -= -1. * np.tan(t/2)*2/T *np.exp(-1. * mu_down) *(-1.)
#(matrix elements between up -> up)
exponent_check[0, 1] += 1 *np.cos(t) *np.exp(-1 * mu_up) *(-1.) *(-1.)
#(matrix elements between down -> down)
exponent_check[3*dim_B , 3*dim_B + 1] += 1 *np.cos(t) *np.exp(-1. * mu_down) *(-1.) *(-1.)

# forward Transpose (antisymm)
exponent_check[3*dim_B +1,0] += 1 * np.tan(t/2)*2/T *np.exp(-1 * mu_up) *(-1.) 
exponent_check[3*dim_B,1] -= 1. * np.tan(t/2)*2/T *np.exp(-1. * mu_down) *(-1.)
exponent_check[1,0] += -1 *np.cos(t) *np.exp(-1. * mu_up) *(-1.) *(-1.)
exponent_check[3*dim_B + 1,3*dim_B] += -1 *np.cos(t) *np.exp(-1. * mu_down) *(-1.) *(-1.)

 
   
exponent_inv = linalg.inv(exponent_check)

print('norm',np.sqrt(linalg.det(exponent_check)))


#store expoent for benchmark 
with h5py.File(filename + ".hdf5", 'w') as f:
    dset_B = f.create_dataset('B', ((dim_B,dim_B)),dtype=np.complex_)
    dset_B[:,:] = B[:,:]

    dset_propag = f.create_dataset('propag', ((2,dim_B//2)),dtype=np.complex_)

    for tau in range (1,dim_B//2):
        dset_propag[0,tau] = exponent_inv[0,3*dim_B -1 -2*tau]
        print('up',dset_propag[0,tau])
        dset_propag[1,tau] = exponent_inv[3*dim_B,2*dim_B -1 -2*tau]
        print('down',dset_propag[1,tau])

