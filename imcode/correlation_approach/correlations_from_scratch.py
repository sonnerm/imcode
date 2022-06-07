from Lohschmidt import Lohschmidt
from evolution_matrix import evolution_matrix
from compute_generators import compute_generators
import numpy as np
import h5py
from create_correlation_block import create_correlation_block
from entropy import entropy
from IM_exponent import IM_exponent
from add_cmplx_random_antisym import add_cmplx_random_antisym
from ham_gs import create_exact_Floquet_ham
from plot_entropy import plot_entropy
from matrix_diag import matrix_diag
from scipy.linalg import expm, logm
from scipy import linalg
from dress_density_matrix import dress_density_matrix
from compute_generators import compute_generators
from Lohschmidt import Lohschmidt
from numpy.linalg import matrix_power
from scipy.linalg import block_diag
from ising_gamma import ising_gamma
import sys
from gm_integral import gm_integral
import os
import math



np.set_printoptions(linewidth=np.nan, precision=1, suppress=True)

# define fixed parameters:
# step sizes for total times t
time_0 = 2
max_time1 = 30
max_time2 = 31
stepsize1 = 1
stepsize2 = 1

time_array = np.append(np.arange(time_0, max_time1, stepsize1) , np.arange(max_time1, max_time2, stepsize2))
print(time_array)

mode =  sys.argv[1] # 'G': compute temporal entanglement entropy from Grassmann approach, 'C': compute temporal entanglement entropy from correlation approach, 'L': compute Lohschmidt echo
write_mode = sys.argv[2] #if the argument is 'w', overwrite file if it exists, otherwise append if it exists
# lattice sites (in the environment):
nsites = int(sys.argv[3]) #total number of sites, including the impurity
# model parameters:
del_t = float(sys.argv[4])
Jx = float(sys.argv[5]) * del_t #* np.pi/2 #0.5# 0.31 # 0.31
Jy =float(sys.argv[6])* del_t# * np.pi/2#np.pi/4+0.3#np.pi/4
g =float(sys.argv[7])* del_t #* np.pi/2#np.pi/4+0.3
init_state = int(sys.argv[8])#0: thermal e^{-\beta XX}, 1: Bell pairs, 2: BCS_GS, 3: Inf. Temp.. Invalied entries will be set to Inf. Temp. (=3)
beta = float(sys.argv[9])#0.4  # temperature
mu_initial_state = float(sys.argv[10])
if mode == 'L':
    g_boundary_mag = float(sys.argv[11])

print('mode', mode)
print('write_mode', write_mode)
print('nsites', nsites)
print('del_t', del_t)
print('Jx', Jx)
print('Jy', Jy)
print('g', g)
print('beta', beta)
print('init_state', init_state)
print('mu_init_state', mu_initial_state)
if mode == 'L':
    print('g_boundary_mag', g_boundary_mag)

iterator = 0 # iterator at beginning of job. If file exist and job is continued, this will be set to the appropraite value below
# total_time = 0 means one floquet-layer

beta_tilde = 0

rho_0_exponent = np.zeros((2 * nsites, 2 * nsites), dtype=np.complex_)
# must be initialized as matrix that diagonalizes dressed density matrix
N_left = np.identity((nsites//2) *2, dtype=np.complex_)
N_right = np.identity((nsites//2) *2, dtype=np.complex_)

nsites_left = nsites // 2
nsites_right =nsites // 2

# find generators and matrices which diagonalize the composite Floquet operator:


G_XY_even_left, G_XY_odd_left, G_g_left, G_1_left = compute_generators(nsites_left, Jx, Jy, g, beta_tilde)
G_XY_even_right, G_XY_odd_right, G_g_right, G_1_right = compute_generators(nsites_right,Jx, Jy, g, beta_tilde)

H_eff_left = -1.j * logm(expm(1.j*G_XY_odd_left) @ expm(1.j*G_XY_even_left))
H_eff_right = -1.j * logm(expm(1.j*G_XY_even_right) @ expm(1.j*G_XY_odd_right))

#print(H_eff_left)
eigenvals_dressed_left, eigenvecs_dressed1 = linalg.eigh(H_eff_left)
eigenvals_dressed_right, eigenvecs_dressed2 = linalg.eigh(H_eff_right)

argsort_left = np.argsort(- eigenvals_dressed_left)
argsort_right = np.argsort(- eigenvals_dressed_right)


#By knowledge of the structure of N, i.e. (N = [[A, B^*],[B, A^*]]), we can construct the right part of the matrix from the left part of the matrix "eigenvecs_dressed", such that all phase factors and the order of eigenvectors are as desired.
N_left = np.identity(H_eff_left.shape[0])
if abs(np.max(eigenvals_dressed_left)) > 1.e-6:
    N_left =   np.bmat([[eigenvecs_dressed1[nsites_left:2*(nsites_left),argsort_left[:nsites_left]].conj(), eigenvecs_dressed1[0:nsites_left,argsort_left[:nsites_left]]],[eigenvecs_dressed1[0:nsites_left,argsort_left[:nsites_left]].conj(),eigenvecs_dressed1[nsites_left:2*(nsites_left),argsort_left[:nsites_left]]]]) 
N_right = np.identity(H_eff_right.shape[0])
if abs(np.max(eigenvals_dressed_right)) > 1.e-6:
    N_right =   np.bmat([[eigenvecs_dressed2[nsites_right:2*(nsites_right),argsort_right[:nsites_right]].conj(), eigenvecs_dressed2[0:nsites_right,argsort_right[:nsites_right]]],[eigenvecs_dressed2[0:nsites_right,argsort_right[:nsites_right]].conj(),eigenvecs_dressed2[nsites_right:2*(nsites_right),argsort_right[:nsites_right]]]]) 

#print(N_left.T.conj() @ H_eff_left @ N_left)  
#print(N_right.T.conj() @ H_eff_right @ N_right)  

n_k_left = np.empty(2*(nsites_left)) 
n_k_left.fill(.5)
n_k_right = np.empty(2*(nsites_right)) 
n_k_right.fill(.5)

Lambda_left = np.diag(n_k_left)
Lambda_right = np.diag(n_k_right)

if init_state == 2:
    n_k_left[:nsites_left] = 1
    n_k_left[nsites_left:] = 0
    n_k_right[:nsites_right] = 1
    n_k_right[nsites_right:] = 0

    Lambda_left = N_left @ np.diag(n_k_left) @ N_left.T.conj()
    Lambda_right = N_right @ np.diag(n_k_right) @ N_right.T.conj()
Lambda_imp = np.empty(2) 
Lambda_imp.fill(.5)

#Lambda = block_diag( Lambda_left[:nsites_left,:nsites_left], 1000*np.diag(Lambda_imp),Lambda_right)
Lambda = block_diag(Lambda_left[:nsites_left,:nsites_left], .5 ,Lambda_right[:nsites_right,:nsites_right], Lambda_left[nsites_left:,nsites_left:], .5 ,Lambda_right[nsites_right:,nsites_right:])
#print(Lambda)
#print(Lambda_left - Lambda_right)
G_XY_even, G_XY_odd, G_g, G_1 = compute_generators(nsites, Jx, Jy, g, beta_tilde)


#coupling to right side
"""
G_XY_even[4,5] = 0
G_XY_even[5,4] = 0
G_XY_even[13,14] = 0
G_XY_even[14,13] = 0

#coupling to left side
G_XY_odd[3,4] = 0
G_XY_odd[4,3] = 0
G_XY_odd[12,13] = 0
G_XY_odd[13,12] = 0
"""
#G_XY_odd = np.zeros(G_XY_even.shape)
#G_XY_even = np.zeros(G_XY_even.shape)
#print(G_XY_odd)
#print(G_XY_even)


evol = (expm(1.j*G_XY_odd) @ expm(1.j*G_XY_even))

print (evol.shape,Lambda.shape)
for i in range (2,6):
    t_2=i
    t_1=0
    print(i, (matrix_power(evol,t_2) @ Lambda @ matrix_power(evol.T.conj(),t_1))[nsites//2,nsites//2])
  