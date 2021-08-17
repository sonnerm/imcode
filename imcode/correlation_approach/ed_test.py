from math import e
from quimb.evo import schrodinger_eq_dop
from scipy.linalg import expm
from scipy.linalg import eigvalsh
from scipy.sparse.linalg import eigsh #diagonalization via Lanczos for initial state
from scipy.linalg import block_diag
import  numpy as np
from scipy import sparse
from quimb import *
np.set_printoptions(linewidth=np.nan, precision=2, suppress=True)

L = 8# number of sites of the spin chain (i.e. INCLUDING THE SYSTEM)

#------------------------------------------------------------ Define initial state density matrix--------------------------------------------------------------------

#--------------- ground state of XY Hamiltonian -----------------------------------------------

#compute initial state (2^(L-1) indices)
#Hamiltonian of XX model:
ham = ham_XY((L - 1), jxy= 1.0, bz = 0.0)#create XY-Hamiltonian using quimb
print (ham)
ham = ham - eye(len(ham)) * 2 * L #shift Hamiltonian by a constant to make sure that eigenvalue with largest magnitude is the ground state
print (ham)

gs_energy, gs = eigsh(ham, 1) #yields ground state vector and corresponding eigenvalue (shifted downwards by 2 * L)

#density matarix for pure ground state of xy-Hamiltonian
state = gs @ gs.T.conj()

#--------------- infinite temperature initial state  -----------------------------------------------

#density matrix for infinite temperature ground state
state = np.identity(2**(L-1)) / 2**(L-1)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------

#Parameters for KIC Floquet evolution
J = 0.31
g = np.pi/4

F_odd = expm(1j * ham_ising(2,0,g)) @ expm(1j * ham_ising(2,J,0))
print('gate_odd\m', F_odd.shape)
F_even = expm(1j * ham_ising(2,J,0))
print('gate_even\m', F_even.shape)

n_traced = 0#this variable keeps track of the number of spins in the spin chain that have been integrated out 

state = np.reshape(state, (2**(L-1) , 1, 1, 2**(L-1)))#reshape density matrix to bring it into general form with open legs



for i in range (int(L/2) - 1):#iteratibely apply Floquet layer and trace out last two spins until the spin chain has become a pure ancilla spin chain.

    
#odd layer
    layer_odd = F_odd
    for i in range (int((L - n_traced)/2) - 1 ):
        layer_odd = kron(layer_odd, F_odd)
    print('layer odd shape',layer_odd.shape)

    #even layer
    layer_even = kron(np.identity(2), F_even)
    for i in range (int((L - n_traced)/2) - 2):
        layer_even = kron(layer_even, F_even)
    print('layer even shape',layer_even.shape)
    layer_even = kron(layer_even, np.identity(2))
  

    F_apply_shape = (2, 2**(L-1 - n_traced), 2, 2**(L-1 - n_traced))#shape needed for layer to multiply it to state
    F = np.reshape(layer_even @ layer_odd, F_apply_shape)
    
    #apply double layer of gates, F
    state = np.einsum('aibj, jqrk, kdle  -> iqabrdel', F, state, F.T.conj())

    #update number of "non-ancilla"-spins of the original chain that will be traced out after this cycle
    n_traced += 2

    #update number of open legs (in spatial direction)
    dim_open_legs_per_branch = 2**n_traced

    #reshape state such that last spin can be traced out
    trace_shape = (2**(L - 1 - n_traced),2 ,2, dim_open_legs_per_branch, dim_open_legs_per_branch, 2**(L - 1 - n_traced), 2, 2)
    state = np.reshape(state, trace_shape)
  
    #trace out the two last spins
    state = np.trace(state, axis1 = 2, axis2 = 7)# trace out last spin
    state = np.trace(state, axis1 = 1, axis2 = 5)# trace out second to last spin   


#last layer consists only of single gate that coupled system and bath (i.e. "odd layer")
F_apply_shape = (2, 2**(L-1 - n_traced), 2, 2**(L-1 - n_traced))#shape needed for layer to multiply it to state

#apply last gate
state = np.einsum('aibj, jqrk, kdle  -> iqabrdel', np.reshape(F_odd, F_apply_shape), state, np.reshape(F_odd.T.conj(), F_apply_shape))

#reshape such that last "non-ancilla" - spin can be traced out. After this step, only open legs in temporal direction remain.
state = np.reshape(state,(2, 2**(2 * L),2))
#trace out last "non-ancilla" - spin
state = np.trace(state, axis1 = 0, axis2 = 2)

state = np.reshape(state, (2**L, 2**L))


entr_eigvals = eigvalsh(state)
print (entr_eigvals)
entropy_values = []
for i in range (L):
    entropy_values.append(-np.sum(np.maximum(entropy_values,0.0)*np.log(np.minimum(np.maximum(entropy_values,1e-32),1.0))))
print (entropy_values)


print('Terminated..')