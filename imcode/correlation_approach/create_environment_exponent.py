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

def find_index(x, tau, bar, t):
    return int(2 * (4*t - 1) * x + 2 * (2*t-1 -tau)-1 + bar)


def create_environment_exponent(init_state, Jx, Jy, g, Jx_boundary, Jy_boundary, g_boundary, mu_initial_state, beta, N_l, t, filename):

  
    N_t = 4 * t 
    nbr_xsi = N_l * 2 * (N_t - 1) #factor 2 since there are two types of grassmanns (with and without bar). This is true if the last and first layers are erased
    

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
    gamma_val_boundary = (1 - tx_boundary * ty_boundary) / T_xy_boundary

    R_quad =  np.dot(1.j,np.array([[-beta_val, alpha_val],[-alpha_val, beta_val]]))
    
    #Matrix that couples only environment variables xsi
    
    A_E = sps.dok_matrix((nbr_xsi, nbr_xsi),dtype=np.complex_)

    #INITIAL STATE 

    #e^-betaZ initial state
    if init_state == 0:
        for x in range (0,N_l):   
            i = find_index(x,0,1,t)
            A_E[i,i+1] += np.exp(2. * beta)
        #normalization_state = ((1. + np.exp(2.*beta) ) / 4)**(N_l)
        #det_factor_state = 1

    #Bell pair initial state
    elif init_state == 1:
        print(beta,'beta')
        for x in range (0,N_l-1,2):   
            i = find_index(x,0,1,t)
            j = i + 2 * (N_t - 1)# equivalent to j = find_index(x + 1,tau,1,t), i.e. i shifted by one to the right in spatial direction
            A_E[i,j] += beta 
            A_E[i+1,j+1] -= beta
        if N_l%2 == 1:#edge spin with unity initial state in case number of sites is odd
            i = find_index(N_l - 1 ,0,1,t)
            A_E[i,i+1] += 1

  
    #general initial state
    elif init_state == 2:
        #DM_compact = compute_Kernel_XX(beta, N_l)
        DM_compact , normalization_state = compute_BCS_Kernel(Jx,Jy,g,mu_initial_state, N_l, filename)
        #det_factor_state = 0.25
        
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

    else:
    #infinite temperature initial state
        for x in range (0,N_l):   
            i = find_index(x,0,1,t)
            A_E[i,i+1] += 1
        #normalization_state = 1#this is  2**N_l / 2**N_l, where the factor 1/ 2**N_l is just a convenient, artifially introduced renormalization so numbers dont become too large..it is precisely cancelled by factor in determinant when IM-matrix element is computed. Real norm of this state is 2**N_l
        #det_factor_state = 0.5
        if init_state > 3:
            print('NO VALID INITIAL STATE SPECIFIED, CONTINUING WITH INF. TEMP.')
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
    
    A_E_norm = A_E.copy()
    
    bound_iter = 0
    for tau in range (2 * t -1, -2 * t, -1):
            if (x_edge_left+tau) % 2 == 1 and tau != 0:
                i = find_index(x_edge_left,tau,1,t)
                A_E_norm[i,i+1] = A_E_norm[i,i+1] *  np.exp(2.j* (g_boundary[bound_iter] - abs(g_boundary[bound_iter])) * np.sign(tau))
                bound_iter += 1
    
    
    A_E = A_E.tocsc()

    A_E += - A_E.T

    return A_E