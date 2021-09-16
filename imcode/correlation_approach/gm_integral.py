from scipy import linalg
import numpy as np
import sys

from scipy import linalg


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


def gm_integral(Jx, Jy, N_l, t):
    N_t = 4 * t 
    nbr_xsi = N_l * 2 * (N_t - 1) #factor 2 since there are two types of grassmanns (with and without bar). This is true if the last and first layers are erased
    nbr_eta = N_t
    #Define coupling parameters
    tx = np.tan(Jx)
    ty = np.tan(Jy)
    T_xy = 1 + tx * ty

    #define prefactors that arise in Grassmann Kernels
    alpha = (tx + ty) / T_xy
    beta = (ty - tx) / T_xy
    gamma = (1 - tx * ty) / T_xy

    R_quad =  np.dot(1.j,np.array([[-beta, alpha],[-alpha, beta]]))
    #print (R_quad)

    #Matrix that couples the system to the first site of the environment
    R = np.zeros((nbr_eta, nbr_xsi),dtype=np.complex_)
    for i in range (2 * t):
        R[2*i:2*i+2, 4*i:4*i+2] = R_quad
    #print (R)

    #Matrix that couples system variables to system variables
    A_s = np.zeros((nbr_eta, nbr_eta))#,dtype=np.complex_)
    for i in range (2 * t):
        A_s[2*i, 2*i+1] = gamma /2
        A_s[2*i+1, 2*i] = - gamma /2
        #print (A_s)

    #Matrix that couples only environment variables xsi
    A_E = np.zeros((nbr_xsi, nbr_xsi),dtype=np.complex_)
    for x in range (0,N_l - 1):#this covers only positive times
        for tau in range (2 * t -1, -2 * t, -1):
            if (x+tau) % 2 == 0 and tau != 0:
                i = find_index(x,tau,1,t)
                j = i + 2 * (N_t - 1)# equivalent to j = find_index(x + 1,tau,1,t), i.e. i shifted by one to the right in spatial direction
                A_E[i:i+2, j:j+2] = np.dot(np.sign(tau),R_quad)#if tau is negative, sign of coupling is switched
                A_E[i,i+1] += gamma
                A_E[j,j+1] += gamma
            elif tau == 0:
                i = find_index(x,0,1,t)
               
                A_E[i,i+1] += 1#infinite temperature initial state

    #for boundary spin on right side, insert identity gate
    x_edge_right = N_l - 1 
    for tau in range (2 * t -1, -2 * t, -1):
            if (x_edge_right+tau) % 2 == 0 and tau != 0:
                i = find_index(x_edge_right,tau,1,t)
                A_E[i,i+1] += 1
    i = find_index(x_edge_right,0,1,t)

    A_E[i,i+1] += 1#infinite temperature initial state at right boundary

    #for boundary spin on right side, insert gamma from gate
    x_edge_left = 0
    for tau in range (2 * t -1, -2 * t, -1):
            if (x_edge_left+tau) % 2 == 1 and tau != 0:
                i = find_index(x_edge_left,tau,1,t)
                A_E[i,i+1] += gamma 

    #measure
    for s in range (N_l):
        for i in range (N_t - 2):
            shift = s * 2 * (N_t - 1)
            A_E[2 * i + 1 + shift,2 * i+2 + shift] += 1

    #boundary condition for measure
    for i in range(N_l):
        A_E[i * 2 * (N_t - 1),(i+1) * 2 * (N_t - 1) -1] += 1


    #antisymmetrize
    for i in range(len(A_E[0])):
        for j in range(i,len(A_E[0])):
            A_E[j,i] = - A_E[i,j]
    #print (A_E)

    B = 2*( A_s + 0.5 * R @ linalg.inv(A_E) @ R.T)
    #print (B)
    return B