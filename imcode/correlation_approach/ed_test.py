from math import e
from os import stat_result
from scipy.linalg import expm
from scipy.linalg import eigvalsh
from scipy.sparse.linalg import eigsh #diagonalization via Lanczos for initial state
from scipy.linalg import block_diag
import  numpy as np
from scipy import sparse
from plot_entropy import plot_entropy

def rdm(vec,sites):
    '''
        Calculates the reduced density matrix of the subsystem defined by ``sites``
    '''
    L=int(np.log2(len(vec)))
    complement=sorted(list(set(range(L))-set(sites)))
    vec=vec/np.sqrt(np.sum(vec.conj()*vec))
    print(sites+complement)
    print(sites)
    vec=vec.reshape((2,)*L)
    vec=np.transpose(vec,sites+complement)
    vec=vec.reshape((2**len(sites),2**(len(complement))))
    ret=np.einsum("ij,kj->ik",vec.conj(),vec)
    return ret

np.set_printoptions(linewidth=np.nan, precision=5, suppress=True)

L = 8# number of sites of the spin chain (i.e. INCLUDING THE SYSTEM)

#------------------------------------------------------------ Define initial state density matrix--------------------------------------------------------------------

#--------------- ground state of XY Hamiltonian -----------------------------------------------

#compute initial state (2^(L-1) indices)
#Hamiltonian of XX model:
"""
ham = ham_XY((L - 1), jxy= 1.0, bz = 0.0)#create XY-Hamiltonian using quimb
print (ham)
ham = ham - eye(len(ham)) * 2 * L #shift Hamiltonian by a constant to make sure that eigenvalue with largest magnitude is the ground state
print (ham)

gs_energy, gs = eigsh(ham, 1) #yields ground state vector and corresponding eigenvalue (shifted downwards by 2 * L)

#density matarix for pure ground state of xy-Hamiltonian
state = gs @ gs.T.conj()
"""
#--------------- infinite temperature initial state  -----------------------------------------------

#density matrix for infinite temperature ground state
state = np.identity(2**(L-1)) / 2**(L-1)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------

entropy_values = np.zeros((L // 2 + 2, L//2 + 2))  # entropies

#Parameters for KIC Floquet evolution
Jx = 0
Jy = 0.7
g = 0.2

sigma_x = [[0,1],[1,0]]
sigma_z = [[1,0],[0,-1]]
ham_Ising = Jy * np.kron(sigma_z, sigma_z)
kick_two_site = g * ( np.kron(sigma_x, np.identity(2)) + np.kron(np.identity(2), sigma_x)  ) 

F_odd = expm(1j * kick_two_site) @ expm(1j * ham_Ising)
F_even = expm(1j * ham_Ising)

n_traced = 0#this variable keeps track of the number of spins in the spin chain that have been integrated out 

#state = np.reshape(state, (2**(L-1) , 1, 1, 2**(L-1)))#reshape density matrix to bring it into general form with open legs
state = np.reshape(state, (2**(L-1) , 1, 2**(L-1)))#reshape density matrix to bring it into general form with open legs

iterator = 0

for i in range (int(L/2) - 1):#iteratively apply Floquet layer and trace out last two spins until the spin chain has become a pure ancilla spin chain.

    
#odd layer
    layer_odd = F_odd
    for i in range (int((L - n_traced)/2) - 1 ):
        layer_odd = np.kron(layer_odd, F_odd)
    print('layer odd shape',layer_odd.shape)
    #even layer
    layer_even = np.kron(np.identity(2), F_even)
    for i in range (int((L - n_traced)/2) - 2):
        layer_even = np.kron(layer_even, F_even)
    print('layer even shape',layer_even.shape)
    layer_even = np.kron(layer_even, np.identity(2))
  

    F_apply_shape = (2, 2**(L-1 - n_traced), 2, 2**(L-1 - n_traced))#shape needed for layer to multiply it to state
    F = np.reshape(layer_even @ layer_odd, F_apply_shape)
    
    #apply double layer of gates, F
    #state = np.einsum('aibj, jqrk, kdle  -> iqabrdel', F, state, F.T.conj())
    state = np.einsum('aibj, jqk, kdle  -> iabqdel', F, state, F.T.conj())

    #update number of "non-ancilla"-spins of the original chain that will be traced out after this cycle
    n_traced += 2

    #update dimensionality of open legs (in temporal direction)
    dim_open_legs_per_branch = 2**n_traced

    #reshape state such that last spin can be traced out
    #trace_shape = (2**(L - 1 - n_traced), 4, dim_open_legs_per_branch, dim_open_legs_per_branch, 2**(L - 1 - n_traced), 4)
    trace_shape = (2**(L - 1 - n_traced), 4, dim_open_legs_per_branch**2, 2**(L - 1 - n_traced), 4)
    state = np.reshape(state, trace_shape)
  
    #trace out the two last spins
    #state = np.trace(state, axis1 = 1, axis2 = 5)# trace out last and second to last spin 
    state = np.trace(state, axis1 = 1, axis2 = 4)# trace out last and second to last spin 


    #compute intermediate temporal entanglement entropy
    if n_traced > 2:
        iterator += 1
        state_for_entr = np.trace(state, axis1 = 0, axis2 = 2)
        for cut in range (max( int(n_traced / 2) - 2, 0 ), int(n_traced / 2) + 1):
            rdm_state = rdm(state_for_entr.reshape(-1), list(range(2 * cut, 2 * (n_traced - cut))))
            entr_eigvals = eigvalsh(rdm_state)
            entropy_values[iterator,0] = int(n_traced / 2)
            entropy_values[iterator,cut] = - np.sum(entr_eigvals * np.log(np.clip(entr_eigvals, 1e-30, 1.0))) 


#last layer consists only of single gate that coupled system and bath (i.e. "odd layer")
F_apply_shape = (2, 2**(L-1 - n_traced), 2, 2**(L-1 - n_traced))#shape needed for layer to multiply it to state

#apply last gate
#state = np.einsum('aibj, jqrk, kdle  -> iqabrdel', np.reshape(F_odd, F_apply_shape), state, np.reshape(F_odd.T.conj(), F_apply_shape))
state = np.einsum('aibj, jqk, kdle  -> iabqdel', np.reshape(F_odd, F_apply_shape), state, np.reshape(F_odd.T.conj(), F_apply_shape))

#reshape such that last "non-ancilla" - spin can be traced out. After this step, only open legs in temporal direction remain.
state = np.reshape(state,(2, 2**(2 * L),2))
#trace out last "non-ancilla" - spin
state = np.trace(state, axis1 = 0, axis2 = 2)

#update number of "non-ancilla"-spins of the original chain that will be traced out after this cycle
n_traced += 2

#state = state.reshape(2**(L//2),2**(L//2),2**(L//2),2**(L//2) )
#rdm_dm = np.trace(state, axis1 = 1, axis2 = 3) # view state as dm and trace out complementary times
iterator += 1

for cut in range (max( int(n_traced / 2) - 2, 0 ),  int(n_traced / 2) + 1):
    rdm_state = rdm(state.reshape(-1), list(range(2 * cut, 2 * (n_traced - cut))))
    entr_eigvals = eigvalsh(rdm_state)
    entropy_values[iterator,0] =  int(n_traced / 2)
    entropy_values[iterator,cut] = - np.sum(entr_eigvals * np.log(np.clip(entr_eigvals, 1e-30, 1.0))) 
print (entropy_values)

plot_entropy(entropy_values, iterator + 1, Jx, Jy, g,  L, 'ED_')

print('Terminated..')