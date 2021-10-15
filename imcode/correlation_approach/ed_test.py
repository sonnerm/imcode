from math import e
from os import stat_result
from numpy.lib.type_check import real
from scipy.linalg import expm, eigh
from scipy.linalg import eigvalsh, eigvals
from scipy.sparse.linalg import eigsh #diagonalization via Lanczos for initial state
import  numpy as np
from scipy import linalg
import sys
from scipy import sparse
from plot_entropy import plot_entropy

def rdm(vec,sites):
    '''
        Calculates the reduced density matrix of the subsystem defined by ``sites``
    '''
    L=int(np.log2(len(vec)))
    complement=sorted(list(set(range(L))-set(sites)))
    vec=vec/np.sqrt(np.sum(vec.conj()*vec))
    vec=vec.reshape((2,)*L)
    vec=np.transpose(vec,sites+complement)
    vec=vec.reshape((2**len(sites),2**(len(complement))))
    ret=np.einsum("ij,kj->ik",vec.conj(),vec)
    return ret

np.set_printoptions(linewidth=np.nan, precision=6, suppress=True)
np.set_printoptions(threshold=sys.maxsize)
#np.set_printoptions(linewidth=470)
L = 9# number of sites of the spin chain (i.e. INCLUDING THE SYSTEM)
beta = 20.0
del_t = 1.
#Parameters for Floquet evolution (can handle KIC as well as XY model)
Jx = 0.5 * del_t 
Jy =  0.3 * del_t #np.pi/4+0.3
g = 0 * del_t #np.pi/4+0.3


#Pauli matrices
sigma_x = np.array([[0,1],[1,0]])
sigma_y = np.array([[0,-1j],[1j,0]])
sigma_z = np.array([[1,0],[0,-1]])

#build two-site Floquet operators
ham_XY = Jx * np.kron(sigma_x, sigma_x) + Jy * np.kron(sigma_y, sigma_y)
kick_two_site = g * ( np.kron(sigma_z, np.identity(2)) + np.kron(np.identity(2), sigma_z)  ) 

F_odd =  expm(1j * ham_XY)
F_even = expm(1j * kick_two_site) @ expm(1j * ham_XY)

#------------------------------------------------------------ Define initial state density matrix--------------------------------------------------------------------


#compute initial state (2^(L-1) indices)



#--------------- XX DM -------------------------------
#Hamiltonian of XX model:
"""
ham_XX = np.zeros((2**(L-1), 2**(L-1)))
ham_XX_twosite = np.real(np.kron(sigma_x,sigma_x) + np.kron(sigma_y,sigma_y) )

for i in range(0,L-2):
    ham_XX += np.kron(np.kron(np.identity(2**(L-3-i)), ham_XX_twosite), np.identity(2**i)) 

#periodic boundary conditions
#ham_XX +=  np.kron( np.kron(sigma_x, np.identity(2**(L-3))), sigma_x) + np.kron( np.kron(sigma_y, np.identity(2**(L-3))), sigma_y) 


state = expm(-beta * ham_XX) #/ np.trace(expm(-beta * ham_XX))
"""
#_____________________________________________________
#Hamiltonian Ground State
#XY-Hamiltonian (effective Floquet)
#dimension of initial state
dim_Hilbert = 2**(L-1)
ham_even = np.zeros((dim_Hilbert,dim_Hilbert),dtype=np.complex_)
ham_odd = np.zeros((dim_Hilbert,dim_Hilbert),dtype=np.complex_)

for i in range (0,L-2, 2):
    print('print even')
    ham_even += np.kron(np.identity(2**i) , np.kron(ham_XY, np.identity(2**(L-3-i))))
for i in range (1,L-2, 2):
    print('print odd')
    ham_odd += np.kron(np.identity(2**i) , np.kron(ham_XY, np.identity(2**(L-3-i))))

commutator = ham_even @ ham_odd - ham_odd @ ham_even
Floquet_ham = ham_even + ham_odd + 0.5j * commutator

ham = Floquet_ham
ham = ham - np.identity(len(ham)) * 2 * L #shift Hamiltonian by a constant to make sure that eigenvalue with largest magnitude is the ground state

gs_energy, gs = eigsh(ham, 1) #yields ground state vector and corresponding eigenvalue (shifted downwards by 2 * L)
print(gs_energy)

#density matrix for pure ground state of xy-Hamiltonian
state = gs @ gs.T.conj()


#--------------- e-beta Z product state DM -------------------------------

"""
one_site = expm(-beta * sigma_z )
state = one_site
for i in range (1,L-1):
    state = np.kron(state,one_site)
"""
#--------------- Bell product state initial DM -------------------------------
"""
Bell = np.zeros((4,4))
Bell[0,0] = 1.
Bell[0,3] = beta
Bell[3,0] = beta
Bell[3,3] = beta**2

sigma_p = sigma_x + 1j * sigma_y
sigma_m = sigma_x - 1j * sigma_y
Bell_comp = np.kron( (np.identity(2)+ sigma_z), (np.identity(2)+ sigma_z)) + beta**2 * np.kron( (np.identity(2)- sigma_z), (np.identity(2)- sigma_z)) + beta * ( np.kron(sigma_p, sigma_p) + np.kron(sigma_m, sigma_m))

print(Bell)
print(Bell_comp)
state = Bell
for i in range (2,L-2,2):
    print('bell:i', i)
    state = np.kron(state, Bell)
if L%2 == 0:
    state = np.kron(state,np.array([[1,0],[0,0]]))

#print('trace', np.trace(state))
"""
#--------------- infinite temperature initial state  (tested) -----------------------------------------------

#density matrix for infinite temperature ground state
#state = np.identity(2**(L-1)) / 2**(L-1)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------


print(state)
entropy_values = np.zeros((L // 2 + 2, L//2 + 2))  # entropies


#if number of sites in environment (L-1) is even, the last spin can directly be traced out

if L%2 == 1:
    #reshape state such that last spin can be traced out
    trace_shape = (2**(L - 2 ), 2, 2**(L - 2 ), 2)
    state = np.reshape(state, trace_shape)
    #trace out the last spin
    state = np.trace(state, axis1 = 1, axis2 = 3)# trace out last and second to last spin 
    L -= 1 #effectively shorten the chain by one spin before floquet layer are applied
    print('updated length to', L ,' (system + environment)')

n_traced = 0#this variable keeps track of the number of spins in the spin chain that have been integrated out 

state = np.reshape(state, (2**(L-1) , 1, 2**(L-1)))#reshape density matrix to bring it into general form with open legs "in the middle"

iterator = 0#this variable counts the number of floquet steps for which the entropy is evaluated. 


for i in range (L//2 - 1):#iteratively apply Floquet layer and trace out last two spins until the spin chain has become a pure ancilla spin chain.

    #odd layer
    layer_odd = F_odd
    for j in range (2,L-1-n_traced,2):
        layer_odd = np.kron(layer_odd, F_odd)

    #even layer
    layer_even = np.identity(2)
    for j in range (1,L-2-n_traced,2):
        layer_even = np.kron(layer_even, F_even)
    layer_even = np.kron(layer_even, np.identity(2))

    F_apply_shape = (2, 2**(L-1 - n_traced), 2, 2**(L-1 - n_traced))#shape needed for layer to multiply it to state
    F = np.reshape(layer_even @ layer_odd, F_apply_shape)
    
    #apply double layer of gates, F
    state = np.einsum('aibj, jqk, eldk  -> iabqdel', F, state, F.conj()) #this is equivalent to np.einsum('aibj, jqk, kdle  -> iabqdel', F, state, F.T.conj())
    
    #update number of "non-ancilla"-spins of the original chain that will be traced out after this cycle
    n_traced += 2

    #update dimensionality of open legs (in temporal direction)
    dim_open_legs_per_branch = 2**n_traced

    #reshape state such that last spin can be traced out
    trace_shape = (2**(L - 1 - n_traced), 4, dim_open_legs_per_branch**2, 2**(L - 1 - n_traced), 4)
    state = np.reshape(state, trace_shape)
  
    #trace out the two last spins
    state = np.trace(state, axis1 = 1, axis2 = 4)# trace out last and second to last spin 


    #compute intermediate temporal entanglement entropy
    if n_traced > 2:#compute temp. ent. entropy only if there is more than one Floquet layer, otherwise it is trivial
        iterator += 1
        state_for_entr = np.trace(state, axis1 = 0, axis2 = 2)#integrate out all spatial sites
        c = int(n_traced / 2)
        for cut in range (c - c%2 , c + 1, 2):#the lower limit can be adjusted depending on how many time cuts one want to compute. if it is set to (c - c%2), only the half-time cut is computed for even final times and the next smaller one for odd final times
            rdm_state = rdm(state_for_entr.reshape(-1), list(range(cut, 2 * n_traced - cut)))#compute reduced density matrix, i.e. interpreting IM as a state and integrating out the corresponding times
            entr_eigvals = eigvalsh(rdm_state)
            entropy_values[iterator,0] = int(n_traced / 2)
            entropy_values[iterator,int(cut / 2)] = - np.sum(entr_eigvals * np.log(np.clip(entr_eigvals, 1e-30, 1.0))) #apply formula to compute entanglement entropy from eigenvalues of reduced density matrix


#last layer consists only of single gate that coupled system and bath (i.e. "odd layer")
F_apply_shape = (2, 2**(L-1 - n_traced), 2, 2**(L-1 - n_traced))#shape needed for layer to multiply it to state

#apply last gate
state = np.einsum('aibj, jqk, eldk  -> iabqdel', np.reshape(F_odd, F_apply_shape), state, np.reshape(F_odd.conj(), F_apply_shape))#this is equivalent to np.einsum('aibj, jqk, kdle  -> iabqdel', np.reshape(F_odd, F_apply_shape), state, np.reshape(F_odd.T.conj(), F_apply_shape))

#reshape such that last "non-ancilla" - spin can be traced out. After this step, only open legs in temporal direction remain.
state = np.reshape(state,(2, 2**(2 * L),2))
#trace out last "non-ancilla" - spin
state = np.trace(state, axis1 = 0, axis2 = 2)

#update number of "non-ancilla"-spins of the original chain that will be traced out after this cycle
n_traced += 2

iterator += 1

print(state.shape)
#print(state.reshape(2,2,2,2,2,2,2,2))

#for comments see above
c = int(n_traced / 2)
for cut in range ( c - c%2 , c + 1, 2):
    rdm_state = rdm(state.reshape(-1), list(range(cut, 2 * n_traced - cut)))
    entr_eigvals = eigvalsh(rdm_state)
    entropy_values[iterator,0] =  int(n_traced / 2)
    entropy_values[iterator,int(cut / 2)] = - np.sum(entr_eigvals * np.log(np.clip(entr_eigvals, 1e-30, 1.0))) 
print (entropy_values)

plot_entropy(entropy_values, iterator + 1, Jx, Jy, g,beta,  L, 'ED_')

print('Successfully terminated..')