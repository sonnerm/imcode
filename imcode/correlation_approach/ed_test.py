from math import e
from quimb.evo import schrodinger_eq_dop
from scipy.linalg import expm
from scipy.sparse.linalg import eigsh #diagonalization via Lanczos for initial state
from scipy.linalg import block_diag
import  numpy as np
from scipy import sparse
from quimb import *
np.set_printoptions(linewidth=np.nan, precision=2, suppress=True)

L = 6 # number of sites of the spin chain (i.e. INCLUDING THE SYSTEM)
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

#density matarix for pure ground state
state = gs @ gs.T.conj()
state = np.reshape(state, (2**(L-1) , 1, 2**(L-1)))

F_odd = expm(1j * ham_ising(2,g)) @ expm(1j * ham_ising(2,J))
print('gate_odd\m', F_odd.shape)
F_even = expm(1j * ham_ising(2,J))
print('gate_even\m', F_even.shape)


n_traced = 0#this variable keeps track of the number of spins in the spin chain that have been integrated out 


for i in range (int(L/2) - 1):#iteratibely apply Floquet layer and trace out last two spins until the spin chain has become a pure ancilla spin chain.

#odd layer
    layer_odd = F_odd
    for i in range (int((L - n_traced)/2) - 1 ):
        layer_odd = kron(layer_odd, F_odd)
    print('layer odd shape',layer_odd.shape)

    #even layer
    layer_even = F_even
    for i in range (int((L - n_traced)/2) - 2):
        layer_even = kron(layer_even, F_even)
    print('layer even shape',layer_even.shape)
    layer_even = kron(layer_even, np.identity(2))
  
    #reshape layer such that it can be multiplied to state
    F = np.einsum('ij, ajbk -> aibk', layer_even , np.reshape(layer_odd, (2, 2**(L-1 - n_traced), 2, 2**(L-1 - n_traced))))

    #apply double layer of gates, F
    state = np.einsum('aibj, jck, eldk  -> iabcdel', F,state, F.conj())

    #update number of open legs (in spatial direction)
    dim_open_legs = 2**(2 * n_traced + 4)

    #reshape state such that last spin can be traced out
    state = np.reshape(state,(2**(L - 1 - n_traced - 2),2 ,2, dim_open_legs, 2**(L - 1 - n_traced - 2), 2, 2))
  
    #trace out the two last spins
    state = np.trace(state, axis1 = 2, axis2 = 6)# trace out last spin
    state = np.trace(state, axis1 = 1, axis2 = 4)# trace out second to last spin

    #update number of "non-ancilla"-spins of the original chain that have been traced out
    n_traced += 2

#last layer consists only of single gate that coupled system and bath (i.e. "odd layer")

#reshape last gate such that it can be mutliplied to state
last_gate =  np.reshape(F_odd, (2,2,2,2))

#apply last gate
state = np.einsum('aibj, jck, eldk  -> iabcdel', last_gate, state, last_gate.conj())

#reshape such that last "non-ancilla" - spin can be traced out. After this step, only open legs in temporal direction remain.
state = np.reshape(state,(2, 2**(2 * L),2))
#trace out last "non-ancilla" - spin
state = np.trace(state, axis1 = 0, axis2 = 2)


state = np.reshape(state, np.tile(2,2 * L))
