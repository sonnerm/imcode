from math import e
from quimb.evo import schrodinger_eq_dop
from scipy.linalg import expm
from scipy.sparse.linalg import eigsh #diagonalization via Lanczos for initial state
from scipy.linalg import block_diag
import  numpy as np
from scipy import sparse
from quimb import *
np.set_printoptions(linewidth=np.nan, precision=2, suppress=True)

L = 6 # number of sites INCLUDING THE SYSTEM, i.e. L = 1 + n_environment
#compute initial state (2^(L-1) indices)
#Hamiltonian of XX model:
ham = ham_XY((L - 1), jxy= 1.0, bz = 0.0)#create XY-Hamiltonian using quimb
print (ham)
ham = ham - eye(len(ham)) * 2 * L #shift Hamiltonian by a constant to make sure that eigenvalue with largest magnitude is the ground state
print (ham)

gs_energy, gs = eigsh(ham, 1) #yields ground state vector and corresponding eigenvalue (shifted downwards by 2 * L)


#Ising interaction J
J = 0.31
g = np.pi/4

print (gs)
#density matarix for pure ground state
state = gs @ gs.T.conj()

F_odd = expm(1j * ham_ising(2,g)) @ expm(1j * ham_ising(2,J))
print('gate_odd\m', F_odd.shape)
F_even = expm(1j * ham_ising(2,J))
print('gate_even\m', F_even.shape)

#odd layer
layer_odd = F_odd
for i in range (int(L/2) - 1):
    print (i)
    layer_odd = kron(layer_odd, F_odd)

#even layer
layer_even = F_even
for i in range (int(L/2 ) - 2):
    print (i)
    layer_even = kron(layer_even, F_even)
print('layer even shape',layer_even.shape)
layer_even = kron(layer_even, np.identity(2))

print('layer even shape',layer_even.shape)
print('layer odd shape',layer_odd.shape)

print (2**(L-1))
print(np.reshape(layer_odd, (2, 2**(L-1), 2, 2**(L-1))).shape)
F = np.einsum('ij, ajbk -> aibk', layer_even , np.reshape(layer_odd, (2, 2**(L-1), 2, 2**(L-1))))

#apply double layer of gates, F
state = np.einsum('aibj, jk, dlck  -> abicdl', F,state, F.conj())
print(state.shape)
n_traced = 0
dim_open_legs = 2**(2 * n_traced + 2)
state = np.reshape(state,(2, 2**(L - 1 - n_traced - 2),2 ,2, dim_open_legs, 2 , 2**(L - 1 - n_traced - 2), 2, 2))
#trace out last spin

print (state.shape)

state = np.trace(state, axis1 = 3, axis2 = 7)# trace out last spin
state = np.trace(state, axis1 = 2, axis2 = 6)# trace out second to last spin
print (state.shape)

state = np.reshape(state, )
