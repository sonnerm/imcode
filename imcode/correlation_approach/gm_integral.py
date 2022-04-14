import numpy as np
import sys
from tests import anti_sym_check
from create_environment_exponent import create_environment_exponent
import h5py
from datetime import datetime
from scipy import linalg
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
def gm_integral(init_state, Jx, Jy,g,mu_initial_state, beta, N_l, t, filename, iterator):
    if init_state == 4:
        print('Three-step initial state not implemented for Grassmanns, switched initial state to Infinite Temperature')
        init_state =3
    #boundary couplings
    Jx_boundary = Jx
    Jy_boundary = Jy
    g_boundary = np.zeros(2*t)#np.array(t*[1] + delta_blip * [-1] + (t-delta_blip)*[1])     #np.zeros(2*t)

    N_t = 4 * t 
    nbr_eta = N_t

    #Define coupling parameters for boundary gate
    tx_boundary = np.tan(Jx_boundary)
    ty_boundary = np.tan(Jy_boundary)
    T_xy_boundary = 1 + tx_boundary * ty_boundary

    print('Number of Floquet (double) layers: ',t)
   #define prefactors that arise in Grassmann Kernels at boundary
    alpha_val_boundary =(tx_boundary + ty_boundary) / T_xy_boundary
    beta_val_boundary = (ty_boundary - tx_boundary) / T_xy_boundary
    gamma_val_boundary = (1 - tx_boundary * ty_boundary) / T_xy_boundary

    R_quad_boundary =  np.dot(1.j,np.array([[-beta_val_boundary, alpha_val_boundary],[-alpha_val_boundary, beta_val_boundary]]))

    
    #Matrix that couples only environment variables xsi
    #A_E = np.zeros((nbr_xsi, nbr_xsi),dtype=np.complex_)
    A_E = create_environment_exponent(init_state,Jx,Jy,g,Jx_boundary,Jy_boundary,g_boundary,mu_initial_state,beta,N_l,t,filename)
    print('size(A_E)', sys.getsizeof(A_E)/ (1024 * 1024))

    #Matrix that couples the system to the first site of the environment
    R = sps.dok_matrix((nbr_eta, 2 * (N_t - 1)),dtype=np.complex_)#store only the part of the matrix that has non-zero entries (in principle it is of size (nbr_eta, nbr_xsi))
    for i in range (t):#2t
        R[2*i:2*i+2, 4*i:4*i+2] = np.dot(1.,R_quad_boundary)#edit np.identity(2)#
    for i in range (t,2*t):#2t
        R[2*i:2*i+2, 4*i:4*i+2] = np.dot(-1.,R_quad_boundary)#edit -1*np.identity(2)

    #Matrix that couples system variables to system variables
    A_s = sps.dok_matrix((nbr_eta, nbr_eta),dtype=np.complex_)
    for i in range (2 * t):
        A_s[2*i, 2*i+1] = gamma_val_boundary
        A_s[2*i+1, 2*i] = - gamma_val_boundary


    identity_matrix = sps.dok_matrix((A_E.shape[1],2 * (N_t - 1)))
    for i in range (2 * (N_t - 1)):
        identity_matrix[i,i] = 1
    
 
    R = R.tocsr()
    A_s = A_s.tocsc()
    identity_matrix = identity_matrix.tocsc()

    now = datetime.now()
    print('Start inversion', now)
    
    #solve for certain columns of inverted matrix A_E:
    A_inv = scipy.sparse.linalg.spsolve(A_E,identity_matrix) #compute only the part of A_inv that is needed

    print('size(A_inv)', sys.getsizeof(A_inv)/ (1024 * 1024), type(A_inv))

    now = datetime.now()
    print('Finish inversion', now)
    #print(A_E.shape)
    #print(A_inv.shape)
    #print(-1j*A_inv[4,2])
    np.set_printoptions(linewidth=np.nan, precision=6, suppress=True)
    #print(A_inv[:2 * (N_t - 1),:].toarray())
    #print(R.toarray())
    B =  (A_s + R  @ A_inv[:2 * (N_t - 1),:] @ R.T).toarray()
   
    print('size(B)', sys.getsizeof(B)/ (1024 * 1024))

    anti_sym_check(B)
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
    """
    #write A_inv and B to file
    with h5py.File(filename + '.hdf5', 'a') as f:
        IM_data = f['IM_exponent']
        #edge_corr_data = f['edge_corr']
        #bulk_corr_data = f['bulk_corr']
        IM_data[iterator,:B.shape[0],:B.shape[0]] = B[:,:]
        #bulk_corr_data[iterator,:len(A_inv[1]),:len(A_inv[0])] = A_inv[:,:]
        #edge_corr_data[iterator,0:2 * (N_t - 1),0:2 * (N_t - 1)] = A_inv.toarray()[0:2 * (N_t - 1),0:2 * (N_t - 1)]
    """

    return B

