
from pickle import TRUE
import numpy as np
from scipy import linalg
import pandas as pd
from create_correlation_block import create_correlation_block
from entropy import entropy
from scipy.linalg import expm
import h5py
from pfapack import pfaffian as pf

def interleaved_gate(D_plus, D_minus):#this function take two gates, one from forward, one from backward branch, and spits out the interleaved gate (16x16)
    gate = np.zeros((16,16),dtype=np.complex_)
    rows = [0,1,4,5,2,3,6,7,8,9,12,13,10,11,14,15]
    for i in range (4):
        for j in range (4):
            gate[rows[i + 4 * j],0] = D_plus[j,0] * D_minus[i,0]
            gate[rows[i + 4 * j],1] = D_plus[j,0] * D_minus[i,1]
            gate[rows[i + 4 * j],2] = D_plus[j,1] * D_minus[i,0]
            gate[rows[i + 4 * j],3] = D_plus[j,1] * D_minus[i,1]
            gate[rows[i + 4 * j],4] = D_plus[j,0] * D_minus[i,2]
            gate[rows[i + 4 * j],5] = D_plus[j,0] * D_minus[i,3]
            gate[rows[i + 4 * j],6] = D_plus[j,1] * D_minus[i,2]
            gate[rows[i + 4 * j],7] = D_plus[j,1] * D_minus[i,3]
            gate[rows[i + 4 * j],8] = D_plus[j,2] * D_minus[i,0]
            gate[rows[i + 4 * j],9] = D_plus[j,2] * D_minus[i,1]
            gate[rows[i + 4 * j],10] = D_plus[j,3] * D_minus[i,0]
            gate[rows[i + 4 * j],11] = D_plus[j,3] * D_minus[i,1]
            gate[rows[i + 4 * j],12] = D_plus[j,2] * D_minus[i,2]
            gate[rows[i + 4 * j],13] = D_plus[j,2] * D_minus[i,3]
            gate[rows[i + 4 * j],14] = D_plus[j,3] * D_minus[i,2]
            gate[rows[i + 4 * j],15] = D_plus[j,3] * D_minus[i,3]
    sign_changes = np.identity(16)
    sign_changes[6,6] *= -1
    sign_changes[7,7] *= -1
    sign_changes[14,14] *= -1
    sign_changes[15,15] *= -1
    
    gate = sign_changes @ gate @ sign_changes

    return gate

def operator_to_kernel(A,string = 1, branch='f',boundary=1):#this function converts an opertor to a grassmann kernel. if variable "branch" is 'b'(backward), make appropriate adjustments for backard branch. if variable "boundary" is "-1", implement antiperiodic boundary conditions
    kernel = np.zeros((4,4),dtype=np.complex_)
    kernel[0,0] = A[0,0]
    kernel[0,1] = - A[1,0]
    kernel[0,2] = A[0,1]
    kernel[0,3] = - A[1,1]
    kernel[1,0] = A[2,0]
    kernel[1,1] = - A[3,0]
    kernel[1,2] = A[2,1]
    kernel[1,3] = - A[3,1]
    kernel[2,0] = - A[0,2]
    kernel[2,1] = - A[1,2]
    kernel[2,2] = A[0,3]
    kernel[2,3] = A[1,3]
    kernel[3,0] = - A[2,2]
    kernel[3,1] = - A[3,2]
    kernel[3,2] = A[2,3]
    kernel[3,3] = A[3,3]

    if branch == 'b':
        backward_order = np.zeros((4,4),dtype=np.complex_)
        backward_order[0,0] = 1
        backward_order[1,2] = 1
        backward_order[2,1] = 1
        backward_order[3,3] = -1
        kernel = backward_order.T @ kernel @ backward_order #take into account that gates on backward branch are written in basis which is in reversed order with respect to forward branch

    if boundary == -1: #antiperiodic boundary conditions (only relevant on forward branch)
        kernel = np.diag([1,-1,1,-1]) @ kernel @ np.diag([1,-1,1,-1])
    
    kernel = np.diag([1,string * 1.j,string * 1.j,1]) @ kernel 
    return kernel

#import exponent of IM

filename = '/Users/julianthoenniss/Documents/PhD/data/compmode=C_o=1_Jx=0.0_Jy=0.0_g=0.0mu=0.0_del_t=1.0_beta=0.0_L=100_init=3_coupling=0.3162'
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
        coeffs_left[i] = coeffs[i] #* np.power(-1.,np.sum(np.arange(nbr_fermions)))

delta_t = 1.
E_d_up = 0.4#0.3
E_d_down = 0.5#0.7#E_d_up
U=0#8. 
t_spinhop = 0.3

H = np.zeros((4,4),dtype=np.complex_)#define time evolution Hamiltonian
#spin hopping
H[2,1] += t_spinhop
H[1,2] += t_spinhop
U_hop = expm(1.j*delta_t *H)#time evolution operator

H = np.zeros((4,4),dtype=np.complex_)#define time evolution Hamiltonian
#Anderson
H[1,1] += E_d_up
H[2,2] += E_d_down
H[3,3] += E_d_up + E_d_down + U

U = expm(1.j*delta_t *H) @ U_hop#time evolution operator


c_up_dag = np.zeros((4,4),dtype=np.complex_) 
c_up_dag[1,0] = 1
c_up_dag[3,2] = -1

c_down_dag = np.zeros((4,4),dtype=np.complex_) 
c_down_dag[2,0] = 1
c_down_dag[3,1] = 1

c_up = np.zeros((4,4),dtype=np.complex_) 
c_up[0,1] = 1
c_up[2,3] = -1

c_down = np.zeros((4,4),dtype=np.complex_) 
c_down[0,2] = 1
c_down[1,3] = 1

print(np.trace(c_up @ U @c_up_dag @ U.T.conj())/np.trace(U @ U.T.conj()))


init_state = np.array([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]])
#temporal boundary condition (antisymm)
temp_bound_plus = np.array([[1, 0, 0, 1],[0,0,0,0],[0,0,0,0],[1,0,0,1]])



print(interleaved_gate(operator_to_kernel(np.eye(4)),operator_to_kernel(np.eye(4))))

print(operator_to_kernel(U),operator_to_kernel(U.T.conj()))

for tau in range (0,nbr_Floquet_layers-1):
    print('tau', tau)

    #spin up propagator
    gate_big_up = operator_to_kernel(c_up,boundary=-1)
    gate_big_down = operator_to_kernel(c_down,boundary=-1)
    gate_big_Z = temp_bound_plus

    for i in range (nbr_Floquet_layers - 2 - tau):
        gate_big_up = np.kron(interleaved_gate(operator_to_kernel(U,string = -1),operator_to_kernel(U.T.conj(),string = -1,branch='b')),gate_big_up)
        gate_big_down = np.kron(interleaved_gate(operator_to_kernel(U,string = -1),operator_to_kernel(U.T.conj(),string = -1,branch='b')),gate_big_down)
        gate_big_Z = np.kron(interleaved_gate(operator_to_kernel(U),operator_to_kernel(U.T.conj(),branch='b')),gate_big_Z)

    gate_big_up = np.kron(interleaved_gate(operator_to_kernel(c_up_dag @ U,string = -1),operator_to_kernel(U.T.conj(),string = -1,branch='b')),gate_big_up)
    gate_big_down = np.kron(interleaved_gate(operator_to_kernel(c_down_dag @ U,string = -1),operator_to_kernel(U.T.conj(),string = -1,branch='b')),gate_big_down)
    gate_big_Z = np.kron(interleaved_gate(operator_to_kernel(U),operator_to_kernel(U.T.conj(),branch='b')),gate_big_Z)
    

    for i in range (tau):
        gate_big_up = np.kron(interleaved_gate(operator_to_kernel(U),operator_to_kernel(U.T.conj(),branch='b')),gate_big_up)
        gate_big_down = np.kron(interleaved_gate(operator_to_kernel(U),operator_to_kernel(U.T.conj(),branch='b')),gate_big_down)
        gate_big_Z = np.kron(interleaved_gate(operator_to_kernel(U),operator_to_kernel(U.T.conj(),branch='b')),gate_big_Z)

    gate_big_up = np.kron(init_state,gate_big_up)
    gate_big_down = np.kron(init_state,gate_big_down)
    gate_big_Z = np.kron(init_state,gate_big_Z)

    Z = np.einsum('i,ij,j->',coeffs,gate_big_Z,coeffs, optimize=True)
    G_up = np.einsum('i,ij,j->',coeffs, gate_big_up,coeffs, optimize=True)
    G_down = np.einsum('i,ij,j->',coeffs, gate_big_down,coeffs, optimize=True)
    print(Z)
    print('up',G_up/Z)
    print('down',G_down/Z)
   