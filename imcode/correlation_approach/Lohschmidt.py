import numpy as np
import sys

from scipy.sparse.csr import csr_matrix
from create_environment_exponent import create_environment_exponent
import h5py
from DM_kernel import compute_Kernel_XX,compute_gate_Kernel, find_index_dm
from datetime import datetime
from scipy.linalg import det
from ham_gs import compute_BCS_Kernel
import multiprocessing as mp
from scipy import linalg
from create_environment_exponent import create_environment_exponent
import scipy.sparse
import scipy.sparse.linalg
import scipy.sparse as sps
#from memory_profiler import profile

now = datetime.now().time() # time object

def func(x, a):
  return a * 1./x

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=470)

def isSparse(array,m, n) :
     
    counter = 0
  
    # Count number of zeros
    # in the matrix
    for i in range(0,m) :
        for j in range(0,n) :
            if (array[i][j] == 0) :
                counter = counter + 1
  
    return (counter >
            ((m * n) // 2))

def find_index(x, tau, bar, t):
    return int(2 * (4*t - 1) * x + 2 * (2*t-1 -tau)-1 + bar)

#@profile
def Lohschmidt(init_state, Jx, Jy,g,mu_initial_state, beta, N_l, t, filename, iterator):

    Jx_boundary = 0
    Jy_boundary = 0
    g_boundary = np.array(t*[1] + t * [-1])   #constant blip of duration t

    A_E = create_environment_exponent(init_state,Jx,Jy,g,Jx_boundary,Jy_boundary,g_boundary,mu_initial_state,beta,N_l,t,filename)
    A_E_norm = A_E.copy().todok()
    
    x_edge_left = 0
    bound_iter = 0
    for tau in range (2 * t -1, -2 * t, -1):
            if (x_edge_left+tau) % 2 == 1 and tau != 0:
                i = find_index(x_edge_left,tau,1,t)
                A_E_norm[i,i+1] = A_E_norm[i,i+1] *  np.exp(2.j* (g_boundary[bound_iter] - abs(g_boundary[bound_iter])) * np.sign(tau))
                A_E_norm[i+1,i] = A_E_norm[i+1,i] *  np.exp(2.j* (g_boundary[bound_iter] - abs(g_boundary[bound_iter])) * np.sign(tau))
                bound_iter += 1
    
    
    A_E_norm = A_E_norm.tocsc()

    
    """
    norm_gate = 1. / (np.cos(Jx)*np.cos(Jy) * (1 + np.tan(Jx)*np.tan(Jy)))
    normalization_circuit = norm_gate**(gate_counter % N_t) #this is the norm such that when divided by this, the unitary circuit gives 1 without eternal legs
    det_factor_circ = norm_gate**(- (gate_counter - gate_counter%N_t) / N_t)
    print('norm_gate', norm_gate)
    print('det_fact_circ', det_factor_circ)
    print('N_circ',normalization_circuit)
    print('N_state',normalization_state)
    N = normalization_state * normalization_circuit
    
    #A_E = np.identity(len(A_E[0]))
    #IM_value = np.sqrt(abs(det( pow(det_factor_state**2, N_l  / nbr_xsi) * pow(det_factor_circ**2, 1. * N_t / nbr_xsi)  * A_E)))/N 
    #IM_value2 = np.sqrt(abs(det( A_E @ A_E_norm_inv  )))
    #print('IM_value:',IM_value )
    """
    A_E_norm_inv = scipy.sparse.linalg.inv(A_E_norm)
    IM_value = np.sqrt(abs(det( (A_E @ A_E_norm_inv).toarray()  )))
    print('IM_value:',IM_value )

    with h5py.File(filename + '.hdf5', 'a') as f:

        const_blip_data = f['const_blip']
        const_blip_data[iterator] = IM_value
       
    print('Lohschmidt echo stored in HDF5 format for t=', t)

