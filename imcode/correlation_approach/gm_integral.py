import numpy as np
import sys

from scipy.sparse.csr import csr_matrix
import h5py
from DM_kernel import compute_Kernel_XX,compute_gate_Kernel, find_index_dm
from datetime import datetime
from scipy.linalg import det
from ham_gs import compute_BCS_Kernel
import multiprocessing as mp
from scipy import linalg
import scipy.sparse
import scipy.sparse.linalg
import scipy.sparse as sps
from joblib import Parallel, delayed
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
def gm_integral(Jx, Jy,g,mu_initial_state, beta, N_l, t, filename, iterator):

    #boundary couplings
    Jx_boundary = Jx
    Jy_boundary = Jy
    delta_blip = t
    g_boundary = np.zeros(2*t)#np.array(t*[1] + delta_blip * [-1] + (t-delta_blip)*[1])     #np.zeros(2*t)

    N_t = 4 * t 
    nbr_xsi = N_l * 2 * (N_t - 1) #factor 2 since there are two types of grassmanns (with and without bar). This is true if the last and first layers are erased
    nbr_eta = N_t


    #Define coupling parametersfor bulk
    tx = np.tan(Jx)
    ty = np.tan(Jy)
    T_xy = 1 + tx * ty


    #Define coupling parameters for boundary gate
    tx_boundary = np.tan(Jx_boundary)
    ty_boundary = np.tan(Jy_boundary)
    T_xy_boundary = 1 + tx_boundary * ty_boundary

    print('Number of Floquet (double) layers: ',t)
    #define prefactors that arise in Grassmann Kernels in bulk
    alpha_val =(tx + ty) / T_xy
    beta_val = (ty - tx) / T_xy
    gamma_val = (1 - tx * ty) / T_xy

    #define prefactors that arise in Grassmann Kernels at boundary
    alpha_val_boundary =(tx_boundary + ty_boundary) / T_xy_boundary
    beta_val_boundary = (ty_boundary - tx_boundary) / T_xy_boundary
    gamma_val_boundary = (1 - tx_boundary * ty_boundary) / T_xy_boundary

    R_quad =  np.dot(1.j,np.array([[-beta_val, alpha_val],[-alpha_val, beta_val]]))
    R_quad_boundary =  np.dot(1.j,np.array([[-beta_val_boundary, alpha_val_boundary],[-alpha_val_boundary, beta_val_boundary]]))

    
    #Matrix that couples only environment variables xsi
    #A_E = np.zeros((nbr_xsi, nbr_xsi),dtype=np.complex_)
  
    A_E = sps.dok_matrix((nbr_xsi, nbr_xsi),dtype=np.complex_)

    #INITIAL STATE 
    #infinite temperature initial state
    #for x in range (0,N_l):   
    #    i = find_index(x,0,1,t)
    #    A_E[i,i+1] += 1
    #normalization_state = 1#this is  2**N_l / 2**N_l, where the factor 1/ 2**N_l is just a convenient, artifially introduced renormalization so numbers dont become too large..it is precisely cancelled by factor in determinant when IM-matrix element is computed. Real norm of this state is 2**N_l
    #det_factor_state = 0.5
    """
    #e^-betaZ initial state
    for x in range (0,N_l):   
        i = find_index(x,0,1,t)
        A_E[i,i+1] += np.exp(2. * beta)
    normalization_state = ((1. + np.exp(2.*beta) ) / 4)**(N_l)
    det_factor_state = 1
    """
    """
    #Bell pair initial state
    print(beta,'beta')
    for x in range (0,N_l-1,2):   
        i = find_index(x,0,1,t)
        j = i + 2 * (N_t - 1)# equivalent to j = find_index(x + 1,tau,1,t), i.e. i shifted by one to the right in spatial direction
        A_E[i,j] += beta 
        A_E[i+1,j+1] -= beta
    if N_l%2 == 1:#edge spin with unity initial state in case number of sites is odd
        i = find_index(N_l - 1 ,0,1,t)
        A_E[i,i+1] += 1
    """
  
    #general initial state
    #DM_compact = compute_Kernel_XX(beta, N_l)
    DM_compact , normalization_state = compute_BCS_Kernel(Jx,Jy,g,mu_initial_state, N_l, filename)
    det_factor_state = 0.25
    
    now = datetime.now()
    print('Start writing', now)
    #integrate into bigger matrix for gm_integral code:
    for x in range (0,N_l):
        for y in range (x,N_l):
            for bar1 in range(2):
                for bar2 in range(2):
                    i = find_index(x,(bar1-1),bar1,t)
                    j = find_index(y,(bar2-1),bar2,t)
                    k = find_index_dm(x,bar1,N_l)
                    l = find_index_dm(y,bar2,N_l)

                    if j>i:
                        A_E[i,j] += DM_compact[k,l]

    gate_counter = 0

    #write in matrix A_E that couples only environment variables
    for x in range (0,N_l - 1):
        for tau in range (2 * t -1, -2 * t, -1):

            if (x+tau) % 2 == 0 and tau != 0:
                i = find_index(x,tau,1,t)
                j = i + 2 * (N_t - 1)# equivalent to j = find_index(x + 1,tau,1,t), i.e. i shifted by one to the right in spatial direction
                A_E[i:i+2, j:j+2] = np.dot(np.sign(tau),R_quad)#if tau is negative, sign of coupling is switched
                A_E[i,i+1] += gamma_val
                A_E[j,j+1] += gamma_val
                gate_counter += 1
                
                if x % 2 == 0 :#with local kicks e^{igZ} after even gates
                    if tau >0:#for tau > 0, the "bar" grassmanns are multiplied with a factor np.exp(-2.j* g)
                        A_E[i,j+1] *= np.exp(-2.j* g)
                        A_E[i,j] *= np.exp(-4.j* g)
                        A_E[i+1,j] *= np.exp(-2.j* g)
                        A_E[i,i+1] *= np.exp(-2.j* g) 
                        A_E[j,j+1] *= np.exp(-2.j* g)

                    else:#for tau < 0, the "non-bar" grassmanns are multiplied with a factor np.exp(+2.j* g)
                        A_E[i,j+1] *= np.exp(2.j* g)
                        A_E[i+1,j+1] *= np.exp(4.j* g)
                        A_E[i+1,j] *= np.exp(2.j* g)
                        A_E[i,i+1] *= np.exp(2.j* g)
                        A_E[j,j+1] *= np.exp(2.j* g)
  
    #for boundary spin on right side, insert identity gate
    x_edge_right = N_l - 1 
    for tau in range (2 * t -1, -2 * t, -1):
            if (x_edge_right+tau) % 2 == 0 and tau != 0:
                i = find_index(x_edge_right,tau,1,t)
               
                if N_l%2 == 1:
                    A_E[i,i+1] += np.exp(-2.j * g* np.sign(tau))
                else: 
                    A_E[i,i+1] += 1
                
    #for boundary spin on left side, insert gamma from gate
    x_edge_left = 0
    bound_iter = 0
    for tau in range (2 * t -1, -2 * t, -1):
            if (x_edge_left+tau) % 2 == 1 and tau != 0:
                i = find_index(x_edge_left,tau,1,t)
                A_E[i,i+1] += gamma_val_boundary * np.exp(-2.j* g_boundary[bound_iter]*np.sign(tau))
                bound_iter += 1
      
    #measure
    for s in range (N_l):
        for i in range (N_t - 2):
            shift = s * 2 * (N_t - 1)
            A_E[2 * i + 1 + shift,2 * i+2 + shift] += 1

    #temporal boundary condition for measure
    for i in range(N_l):
        A_E[i * 2 * (N_t - 1),(i+1) * 2 * (N_t - 1) -1] += 1
   
    now = datetime.now()
    print('End writing', now)
    """
    A_E_norm = A_E.copy()
    
    bound_iter = 0
    for tau in range (2 * t -1, -2 * t, -1):
            if (x_edge_left+tau) % 2 == 1 and tau != 0:
                i = find_index(x_edge_left,tau,1,t)
                A_E_norm[i,i+1] = A_E_norm[i,i+1] *  np.exp(2.j* (g_boundary[bound_iter] - abs(g_boundary[bound_iter])) * np.sign(tau))
                bound_iter += 1
    """
    

    #Matrix that couples the system to the first site of the environment
    R = sps.dok_matrix((nbr_eta, 2 * (N_t - 1)),dtype=np.complex_)#store only the part of the matrix that has non-zero entries (in principle it is of size (nbr_eta, nbr_xsi))
    for i in range (t):#2t
        R[2*i:2*i+2, 4*i:4*i+2] = np.dot(1.,R_quad_boundary)
    for i in range (t,2*t):#2t
        R[2*i:2*i+2, 4*i:4*i+2] = np.dot(-1.,R_quad_boundary)

    #Matrix that couples system variables to system variables
    A_s = sps.dok_matrix((nbr_eta, nbr_eta),dtype=np.complex_)
    for i in range (2 * t):
        A_s[2*i, 2*i+1] = gamma_val_boundary
        A_s[2*i+1, 2*i] = - gamma_val_boundary
 

    """
    norm_gate = 1. / (np.cos(Jx)*np.cos(Jy) * (1 + np.tan(Jx)*np.tan(Jy)))
    normalization_circuit = norm_gate**(gate_counter % N_t) #this is the norm such that when divided by this, the unitary circuit gives 1 without eternal legs
    det_factor_circ = norm_gate**(- (gate_counter - gate_counter%N_t) / N_t)
    print('norm_gate', norm_gate)
    print('det_fact_circ', det_factor_circ)
    print('N_circ',normalization_circuit)
    print('N_state',normalization_state)
    N = normalization_state * normalization_circuit
    
   
    #A_E_norm_inv = linalg.inv(A_E_norm)
    #A_E = np.identity(len(A_E[0]))
    #IM_value = np.sqrt(abs(det( pow(det_factor_state**2, N_l  / nbr_xsi) * pow(det_factor_circ**2, 1. * N_t / nbr_xsi)  * A_E)))/N 
    #IM_value2 = np.sqrt(abs(det( A_E @ A_E_norm_inv  )))
    #print('IM_value:',IM_value )

    #print('IM_value2:',IM_value2 )
    """

    print('size(A_E)', sys.getsizeof(A_E)/ (1024 * 1024))

    identity_matrix = sps.dok_matrix((A_E.shape[1],2 * (N_t - 1)))
    for i in range (2 * (N_t - 1)):
        identity_matrix[i,i] = 1
    
    A_E = A_E.tocsc()
    R = R.tocsr()
    A_s = A_s.tocsc()
    identity_matrix = identity_matrix.tocsc()

    A_E += - A_E.T


    #num_cores = mp.cpu_count()
    #if __name__ == "__main__":
    #    A_inv[:,0:2 * (N_t - 1)] = (Parallel(n_jobs=num_cores)(delayed(scipy.sparse.linalg.spsolve)(A_E_sparse, identity_matrix[:,i]) for i in range(2 * (N_t - 1)) )).toarray().T
   
    now = datetime.now()
    print('Start inversion', now)
    
    #solve for certain columns of inverted matrix A_E:
    A_inv = scipy.sparse.linalg.spsolve(A_E,identity_matrix) #compute only the part of A_inv that is needed

    print('size(A_inv)', sys.getsizeof(A_inv)/ (1024 * 1024), type(A_inv))

    now = datetime.now()
    print('Finish inversion', now)

    B =  (A_s  +  R  @ A_inv[:2 * (N_t - 1),:] @ R.T).toarray()
    print('size(B)', sys.getsizeof(B)/ (1024 * 1024))

    """
    #compare to result without sparse matrices
    identity_matrix_comp = np.identity(A_E.shape[0])
    A_inv_comp = np.zeros(A_E.shape,dtype=np.complex_)
    A_E_comp = A_E.toarray()
    A_E_comp += - A_E_comp.T
    A_inv_comp[:,0:2 * (N_t - 1)] = np.linalg.solve(A_E_comp,identity_matrix_comp[:,0:2 * (N_t - 1)]) 
    A_inv_compare = A_inv_comp - A_inv.toarray()

    counter = 0
    for i in range(A_inv.shape[0]):
        for j in range(A_inv.shape[1]):
            counter += abs(A_inv_compare[i,j])
    print('compare A_E_inv', counter)

    B_comp = A_s.toarray() + R.toarray()  @ A_inv_comp @ R.toarray().T
    counter = 0
    B_comp -= B
    for i in range(B_comp.shape[0]):
        for j in range(B_comp.shape[1]):
            counter += abs(B_comp[i,j])
    print('compare B', counter)
    """


 
    #write A_inv and B to file
    with h5py.File(filename + '.hdf5', 'a') as f:
        IM_data = f['IM_exponent']
        edge_corr_data = f['edge_corr']
        #bulk_corr_data = f['bulk_corr']
        const_blip_data = f['const_blip']
        IM_data[iterator,:B.shape[0],:B.shape[0]] = B[:,:]
        #bulk_corr_data[iterator,:len(A_inv[1]),:len(A_inv[0])] = A_inv[:,:]
        #edge_corr_data[iterator,0:2 * (N_t - 1),0:2 * (N_t - 1)] = A_inv.toarray()[0:2 * (N_t - 1),0:2 * (N_t - 1)]
        #const_blip_data[iterator] = IM_value
       
    
    """
    #compare to standard inversion 
    A_inv_comp = linalg.inv(A_E)
    B_comp =  A_s +  R @ A_inv_comp @ R.T
    abs=0.
    for i in range(len(B[0])):
        for j in range(len(B[0])):
            abs += np.abs(B[i,j] - B_comp[i,j])
    print('abs',abs)
    """
 
    return B

