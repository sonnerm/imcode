from numpy.core.numeric import identity
from scipy import linalg
import numpy as np
import sys
import scipy
from scipy import linalg
from DM_kernel import compute_Kernel_XX, find_index_dm
from datetime import datetime

now = datetime.now().time() # time object


np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=470)

"""
l = 2
t = 2 #nbr of double layer applications
N_t = 4 * t
N_l = 2 * l #nbr of spins in initial state
"""

def find_index(x, tau, bar, t):
    return int(2 * (4*t - 1) * x + 2 * (2*t-1 -tau)-1 + bar)


def gm_integral(Jx, Jy,g,beta, N_l, t):
    N_t = 4 * t 
    nbr_xsi = N_l * 2 * (N_t - 1) #factor 2 since there are two types of grassmanns (with and without bar). This is true if the last and first layers are erased
    nbr_eta = N_t
    #Define coupling parameters
    tx = np.tan(Jx)
    ty = np.tan(Jy)
    T_xy = 1 + tx * ty
    print('Number of Floquet (double) layers: ',t)
    #define prefactors that arise in Grassmann Kernels
    alpha_val =(tx + ty) / T_xy
    beta_val = (ty - tx) / T_xy
    gamma_val = (1 - tx * ty) / T_xy

    R_quad =  np.dot(1.j,np.array([[-beta_val, alpha_val],[-alpha_val, beta_val]]))
    #print (R_quad)
    print('t1')
    #Matrix that couples the system to the first site of the environment
    R = np.zeros((nbr_eta, nbr_xsi),dtype=np.complex_)
    for i in range (t):#2t
        R[2*i:2*i+2, 4*i:4*i+2] = np.dot(1.,R_quad)
    for i in range (t,2*t):#2t
        R[2*i:2*i+2, 4*i:4*i+2] = np.dot(-1.,R_quad)
    #print('R')
    #print (R)
   

    #Matrix that couples system variables to system variables
    A_s = np.zeros((nbr_eta, nbr_eta),dtype=np.complex_)
    for i in range (2 * t):
        A_s[2*i, 2*i+1] = gamma_val
        A_s[2*i+1, 2*i] = - gamma_val
    #print('AS')
    #print (A_s)
    
    #Matrix that couples only environment variables xsi
    A_E = np.zeros((nbr_xsi, nbr_xsi),dtype=np.complex_)
    for x in range (0,N_l - 1):
        for tau in range (2 * t -1, -2 * t, -1):
            if (x+tau) % 2 == 0 and tau != 0:
                i = find_index(x,tau,1,t)
                j = i + 2 * (N_t - 1)# equivalent to j = find_index(x + 1,tau,1,t), i.e. i shifted by one to the right in spatial direction
                A_E[i:i+2, j:j+2] = np.dot(np.sign(tau),R_quad)#if tau is negative, sign of coupling is switched
                A_E[i,i+1] += gamma_val
                A_E[j,j+1] += gamma_val
                
                if x % 2 == 0 :#with local kicks e^{igZ} after even gates
                    if tau >0:#for tau > 0, , the "bar" grassmanns are multiplied with a factor np.exp(-2.j* g)
                        A_E[i,j+1] *= np.exp(-2.j* g)
                        A_E[i,j] *= np.exp(-4.j* g)
                        A_E[i+1,j] *= np.exp(-2.j* g)
                        A_E[i,i+1] *= np.exp(-2.j* g)
                        A_E[j,j+1] *= np.exp(-2.j* g)

                    else:#for tau > 0, , the "non-bar" grassmanns are multiplied with a factor np.exp(+2.j* g)
                        A_E[i,j+1] *= np.exp(2.j* g)
                        A_E[i+1,j+1] *= np.exp(4.j* g)
                        A_E[i+1,j] *= np.exp(2.j* g)
                        A_E[i,i+1] *= np.exp(2.j* g)
                        A_E[j,j+1] *= np.exp(2.j* g)

                
    #initial state
    
    #infinite temperature initial state
    #for x in range (0,N_l):   
    #    i = find_index(x,0,1,t)
    #    A_E[i,i+1] += 1
    
    
    #e^-betaZ initial state
    #for x in range (0,N_l):   
    #    i = find_index(x,0,1,t)
    #    A_E[i,i+1] += np.exp(2. * beta)
        #A_E[i,i+1] += np.exp(2. * beta*x*0.125)
    
    
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
    DM_compact = compute_Kernel_XX(beta, N_l)
    
    #integrate into bigger matrix for gm_integral code:
    for x in range (0,N_l):
        for y in range (x,N_l):
            for bar1 in range(2):
                for bar2 in range(2):
                    i = find_index(x,(bar1-1),bar1,t)
                    j = find_index(y,(bar2-1),bar2,t)
                    k = find_index_dm(x,bar1,N_l)
                    l = find_index_dm(y,bar2,N_l)
                    if j>=i:
                        A_E[i,j] += DM_compact[k,l]
                        A_test[i,j] += DM_compact[k,l]
    
   

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
    for tau in range (2 * t -1, -2 * t, -1):
            if (x_edge_left+tau) % 2 == 1 and tau != 0:
                i = find_index(x_edge_left,tau,1,t)
                A_E[i,i+1] += gamma_val

    #measure
    for s in range (N_l):
        for i in range (N_t - 2):
            shift = s * 2 * (N_t - 1)
            A_E[2 * i + 1 + shift,2 * i+2 + shift] += 1

    #temporal boundary condition for measure
    for i in range(N_l):
        A_E[i * 2 * (N_t - 1),(i+1) * 2 * (N_t - 1) -1] += 1

    #antisymmetrize
    for i in range(len(A_E[0])):
        for j in range(i,len(A_E[0])):
            A_E[j,i] = - A_E[i,j]


    #add local gates


    #solve for certain columns of inverted matrix A_E:
    #A_inv = np.zeros(A_E.shape)
   
    #identity_matrix = np.identity(len(A_E[0]))
   
    #A_inv[:,0:4 * (N_t - 1)] = np.linalg.solve(A_E,identity_matrix[:,0:4 * (N_t - 1)])

    B =  A_s +  R @ linalg.inv(A_E) @ R.T
    
    """
    #compare to standard inversion 
    B_comp =  A_s +  R @ A_inv_comp @ R.T
    abs=0.
    for i in range(len(B[0])):
        for j in range(len(B[0])):
            abs += np.abs(B[i,j] - B_comp[i,j])
    print('abs',abs)
    """

    return B

