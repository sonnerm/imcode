
from pickle import TRUE
import numpy as np
from scipy import linalg
import pandas as pd
from create_correlation_block import create_correlation_block
from entropy import entropy
import h5py
from pfapack import pfaffian as pf


#import exponent of IM

filename = '/Users/julianthoenniss/Documents/PhD/data/compmode=C_o=1_Jx=1.0_Jy=1.0_g=0.0mu=0.0_del_t=0.1_beta=0.0_L=10_init=3_coupling=1.0_t1-5'
iter = 2

#if exponent B is read out
with h5py.File(filename + '.hdf5', 'r') as f:
    times_read = f['temp_entr']
    nbr_Floquet_layers  = int(times_read[iter,0])
    print('times: ', nbr_Floquet_layers, ' iteration: ', iter )

with h5py.File(filename + '.hdf5', 'r') as f:
    B = f['IM_exponent'][iter,:4*nbr_Floquet_layers,:4*nbr_Floquet_layers] 

S = np.zeros(B.shape,dtype=np.complex_)
for i in range (B.shape[0]//4):#order plus and minus next to each other
    S [B.shape[0] // 2 - (2 * i) - 2,4 * i] = 1
    S [B.shape[0] // 2 - (2 * i) - 1,4 * i + 2] = 1
    S [B.shape[0] // 2 + (2 * i) ,4 * i + 1] = 1
    S [B.shape[0] // 2 + (2 * i) + 1,4 * i + 3] = 1

B = S @ B @ S.T

#the following two transformation bring it into in/out- basis (not theta, zeta)
rot = np.zeros(B.shape)
for i in range(0,4*nbr_Floquet_layers, 2):#go from bar, nonbar to zeta, theta
    rot[i,i] = 1./np.sqrt(2)
    rot[i,i+1] = 1./np.sqrt(2)
    rot[i+1,i] = - 1./np.sqrt(2) * np.sign(2*nbr_Floquet_layers - i-1)
    rot[i+1,i+1] = 1./np.sqrt(2) * np.sign(2*nbr_Floquet_layers - i-1)
B = rot.T @ B @ rot


#here, the matrix B is in the Grassmann convention
# adjust signs that make the influence matrix a vectorized state
for i in range (B.shape[0]):
    for j in range (B.shape[0]):
        if (i+j)%2 == 1:
            B[i,j] *=-1

print(B)

U = np.zeros(B.shape)#order in the way specified in pdf for him (forward,backward,forward,backward,...)
for i in range (B.shape[0]//4):
    U[4*i, B.shape[0] //2 - (2*i) -1] = 1
    U[4*i + 1, B.shape[0] //2 + (2*i)] = 1
    U[4*i + 2, B.shape[0] //2 - (2*i) -2] = 1
    U[4*i + 3, B.shape[0] //2 + (2*i) + 1] = 1
B = U @ B @ U.T 


#The coefficients of the many body wavefunction are given by the slaterdeterminant
len_IM = 2**B.shape[0] # this is the number of entries of the IM viewed as a MB-wavefunction
max_binary = np.binary_repr(len_IM - 1)# binary string that encodes the maximal index we want to encode
coeffs = np.zeros((len_IM), dtype=np.complex_) # array that will store the components of the wavefunction in binary order
coeffs_left = np.zeros((len_IM), dtype=np.complex_) # array that will store the components of the wavefunction in binary order

coeffs[0] = 1
coeffs_left[0] = 1
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
        coeffs_left[i] = coeffs[i] * np.power(-1.,np.sum(np.arange(nbr_fermions)))
print(coeffs)
print(coeffs_left)

delta_t = 0.1
E_d_up = -6.
E_d_down = E_d_up
U=12.
alpha = np.exp(1.j * delta_t * E_d_up)
beta = np.exp(1.j * delta_t * E_d_down)
gamma = np.exp(1.j * delta_t * (E_d_up + E_d_down)) * np.exp(1.j * delta_t * U)


gate = np.zeros((16,16),dtype=np.complex_)
gate[0,0] = 1
gate[0,5] = -alpha.conj()
gate[5,0] = beta.conj()
gate[5,5] = - gamma.conj()
gate[0,10] = alpha
gate[0,15] =  alpha * alpha.conj()
gate[5,10] =  alpha * beta.conj()
gate[5,15] = alpha * gamma.conj()
gate[10,0] = -beta
gate[10,5] =  beta * alpha.conj()
gate[15,0] =  beta * beta.conj()
gate[15,5] =  -beta * gamma.conj()
gate[10,10] = - gamma
gate[10,15] = -gamma * alpha.conj()
gate[15,10] =  gamma * beta.conj()
gate[15,15] = gamma * gamma.conj()


inf_temp_init_state = .25* np.array([[1, 0, 0, -1],[0,0,0,0],[0,0,0,0],[1,0,0,-1]])
up_full_init_state = .5 * np.array([[1, 0, 0, 0],[0,0,0,0],[0,0,0,0],[1,0,0,0]])
init_state = up_full_init_state
print(init_state)

#temporal boundary condition (antisymm)
temp_bound = np.array([[1, 0, 0, -1],[0,0,0,0],[0,0,0,0],[1,0,0,-1]])


gate_big = np.kron(init_state, gate)
for i in range (nbr_Floquet_layers - 2):
    gate_big = np.kron(gate_big,gate)

gate_big_partitionsum = np.kron(gate_big, temp_bound)
Z = np.einsum('i,ij,j->',coeffs_left,gate_big_partitionsum,coeffs, optimize=True)
print(Z)

gate_big = np.kron(gate_big,gate)
gate_big = gate_big.reshape(16**nbr_Floquet_layers, 4, 16**nbr_Floquet_layers, 4)
DM = np.einsum('i,ikjl,j->kl',coeffs_left, gate_big,coeffs, optimize=True)
print(DM)
print(DM/Z)

norm = np.einsum('i,i->',coeffs,coeffs, optimize=True)
print(norm)
