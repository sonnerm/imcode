
from pickle import TRUE
import numpy as np
from scipy import linalg
import pandas as pd
from create_correlation_block import create_correlation_block
from entropy import entropy
import h5py
from pfapack import pfaffian as pf


#import exponent of IM

filename = '/Users/julianthoenniss/Documents/PhD/data/B_imag'

delta_t = 0.1

#if exponent B is read out
with h5py.File(filename + '.hdf5', 'r') as f:
    B_data = f['B']
    B = B_data[:,:]
    dim_B = B.shape[0]
print(B.shape)
print(B)


# creating an empty array to store the reversed matrix
B = B[::-1,::-1]
print(B)
#reorder entries such that input at time zero becomes last entry:
B_reshuf = np.zeros((dim_B,dim_B),dtype=np.complex_)
B_reshuf[:dim_B-1,:dim_B-1] = B[1:dim_B,1:dim_B]
B_reshuf[dim_B-1,:dim_B-1] = B[0,1:dim_B]
B_reshuf[:dim_B-1,dim_B-1] = B[1:dim_B,0]
B_reshuf[dim_B-1,dim_B-1] = B[0,0]
B = B_reshuf


#The coefficients of the many body wavefunction are given by the slaterdeterminant
len_IM = 2**B.shape[0] # this is the number of entries of the IM viewed as a MB-wavefunction
max_binary = np.binary_repr(len_IM - 1)# binary string that encodes the maximal index we want to encode
coeffs = np.zeros((len_IM), dtype=np.complex_) # array that will store the components of the wavefunction in binary order
coeffs_left = np.zeros((len_IM), dtype=np.complex_) # array that will store the components of the wavefunction in binary order

coeffs[0] = 1
for i in range (1,len_IM):#index goes through the elements of the wavefunction indices
    #determine the binary representation of the wavefunction index:
    #This tells us which rows and columns of B we need to compute the prefactor
    binary_index = np.binary_repr(i).zfill(len(max_binary)) #zfill fills zeros from the left to binary representations such that all of them have the same lentgh, set by the maximal index we want to encode.
    
    read_out_grid = []#in this grid, we store the indexes for rows/columns that we want to extract to compute a certain element
    for j in range(len(max_binary)):#little function to extract the positions of zeros and ones in a given binary string
        if binary_index[j] == '1':
            read_out_grid.append(j)

    nbr_fermions = len(read_out_grid)
  
    if nbr_fermions%2 == 0:
        B_sub = np.array(pd.DataFrame(B).iloc[read_out_grid,read_out_grid]) #convert numpy array of B to grid such that we can extract submatrix by index, then convert back to numpy array
        coeffs[i] = pf.pfaffian(B_sub)#the wavefunction coefficients are given by the pfaffian of the submatrix
print(coeffs)
"""delta_t = 0.1
E_d_up = -4. 
E_d_down = E_d_up
U=8. 
alpha = np.exp(1.j * delta_t * E_d_up)
beta = np.exp(1.j * delta_t * E_d_down)
gamma = np.exp(1.j * delta_t * (E_d_up + E_d_down)) * np.exp(1.j * delta_t * U)

sign_overlap = -1
sign_gm_order = -1
#additional signs here from defining the signs from the overlap into the MPO
gate = np.zeros((16,16),dtype=np.complex_)
gate[0,0] = 1
gate[0,5] = -alpha.conj() * sign_overlap
gate[5,0] = beta.conj() * sign_overlap
gate[5,5] = - gamma.conj() 
gate[0,10] = alpha  * sign_overlap
gate[0,15] =  alpha * alpha.conj()
gate[5,10] =  alpha * beta.conj() 
gate[5,15] = alpha * gamma.conj()  * sign_overlap
gate[10,0] = -beta  * sign_overlap
gate[10,5] =  beta * alpha.conj() 
gate[15,0] =  beta * beta.conj()
gate[15,5] =  -beta * gamma.conj()  * sign_overlap
gate[10,10] = - gamma 
gate[10,15] = -gamma * alpha.conj()  * sign_overlap
gate[15,10] =  gamma * beta.conj() * sign_overlap
gate[15,15] = gamma * gamma.conj() 

#account for ordering or Grassmanns for 'bra' (the columns that add an odd number of grassmann pairs to the state)
#this relies on the fact that the IM is even, such that only even components of impurity gate contribute. If this is not the case, sign must be changed on level of full many body wavefcuntion (uncomment factor for coeffs_left)
#multiply each row by exp(i * pi/2 * n), where n is the number of Grassmann variables for the down spin.
if sign_gm_order == -1:
    #one down Grassmanns:
    gate[[1,2,4,8],:] *= np.exp(1.j * np.pi/2)
    #two down Grassmanns:
    gate[[3,5,6,9,10,12],:] *= np.exp(1.j * np.pi)
    #three down Grassmanns:
    gate[[7,11,13,14],:] *= np.exp(1.j * 3./2 * np.pi )
"""

#Spin hopping
t=0.0#1.3 * delta_t
E_d_up =0.52# 0.3 * delta_t
E_d_down =0.52# E_d_up 

alpha1 = -2.*np.tan(t/2)/(1 + np.tan(t/2)**2) * np.exp(-1. * E_d_up)
alpha2 = -2.*np.tan(t/2)/(1 + np.tan(t/2)**2) * np.exp(-1. * E_d_down)
beta = np.cos(t) * np.exp(-1. * E_d_up)
gamma = np.cos(t) * np.exp(-1. * E_d_down)
omega = - alpha1 * alpha2 + beta * gamma


gate = np.zeros((4,4),dtype=np.complex_)
gate[0,0] = 1
gate[2,1] = - alpha1
gate[1,2] = alpha2
gate[0,3] = - beta 
gate[3,0] = - gamma 
gate[3,3] = omega 

print(gate)
#antiperiodic bc, no operator
gate_boundary = gate.copy()
gate_boundary[1,:] *= -1.
gate_boundary[3,:] *= -1.
gate_boundary[:,1] *= -1.
gate_boundary[:,3] *= -1.
print(gate_boundary)

gate_big = gate_boundary
for i in range (dim_B//2-1):
    gate_big = np.kron(gate,gate_big)
Z = np.einsum('i,ij,j->',coeffs,gate_big,coeffs, optimize=True) 
print('Z',Z)
print('big_trace_up',gate_big @ coeffs  )
print('big_trace_down',coeffs  @ gate_big)

#store expoent for benchmark 
with h5py.File(filename + "_ED.hdf5", 'w') as f:
    dset_propag = f.create_dataset('propag', ((2,dim_B//2)),dtype=np.complex_)


#spin up propagator
gate_boundary_cdag_up = np.zeros((4,4),dtype=np.complex_)
gate_boundary_cdag_up[0,1] = 1  
gate_boundary_cdag_up[3,1] = np.exp(-E_d_up) 

gate_c_up= np.zeros((4,4),dtype=np.complex_)
gate_c_up[0,2] = np.exp(-E_d_up)  
gate_c_up[3,2] = -np.exp(-(2*E_d_up)) 

print('coeffs',coeffs)
for tau in range (dim_B//2-1):
    print('tau',tau)
    gate_big = gate_boundary_cdag_up
    for i in range (dim_B//2-1 - tau - 1):
        print('1',i,gate_big.shape)
        gate_big = np.kron(gate,gate_big)
    gate_big = np.kron(gate_c_up,gate_big)
    print('2',gate_big.shape)
    for i in range (tau):
        print(i)
        print('3',i,gate_big.shape)
        gate_big = np.kron(gate,gate_big)

    #print(coeffs.shape)
    print('big_up',gate_big @ coeffs)
    G = np.einsum('i,ij,j->',coeffs,gate_big,coeffs, optimize=True) 
    print('propag_up',G/Z)
    with h5py.File(filename + "_ED.hdf5", 'a') as f:
        dset_propag = f['propag']
        dset_propag[0,tau+1] = G/Z

#spin down propagator
gate_boundary_cdag_down = np.zeros((4,4),dtype=np.complex_)
gate_boundary_cdag_down[1,0] = -1 
gate_boundary_cdag_down[1,3] = -1 * np.exp(-E_d_down) 

gate_c_down= np.zeros((4,4),dtype=np.complex_)
gate_c_down[2,0] = -1.* np.exp(-E_d_down) 
gate_c_down[2,3] = 1. * np.exp(-(2*E_d_down)) 

for tau in range (dim_B//2-1):
    print('tau',tau)
    gate_big = gate_boundary_cdag_down
    for i in range (dim_B//2-1 - tau - 1):
        print('1',i,gate_big.shape)
        gate_big = np.kron(gate,gate_big)

    gate_big = np.kron(gate_c_down,gate_big)
    print('2',gate_big.shape)
    for i in range (tau):
        print(i)
        print('3',i,gate_big.shape)
        gate_big = np.kron(gate,gate_big)
    
    print('big_down',coeffs@gate_big )

    #print(coeffs.shape)
    #print(gate_big.shape)
    G = np.einsum('i,ij,j->',coeffs,gate_big,coeffs, optimize=True) 
    print('propag_down',G/Z)
    with h5py.File(filename + "_ED.hdf5", 'a') as f:
        dset_propag = f['propag']
        dset_propag[1,tau+1] = G/Z
