from heapq import _heapify_max
from Lohschmidt import Lohschmidt
from evolution_matrix import evolution_matrix
from compute_generators import compute_generators
import numpy as np
import h5py
import matplotlib.pyplot as plt
from create_correlation_block import create_correlation_block
from entropy import entropy
from IM_exponent import IM_exponent
from add_cmplx_random_antisym import add_cmplx_random_antisym
from ham_gs import create_exact_Floquet_ham
from plot_entropy import plot_entropy
from matrix_diag import matrix_diag
from scipy.linalg import expm, logm, sqrtm
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
time_0 = 400
max_time1 =401
max_time2 = 401
stepsize1 = 100
stepsize2 = 5

time_array = np.concatenate((np.arange(time_0, max_time1, stepsize1) , np.arange(max_time1, max_time2, stepsize2)))
time_array = np.arange(20,21,1)
print(time_array)
#time_array = np.array([400])

mode =  sys.argv[1] # 'G': compute temporal entanglement entropy from Grassmann approach, 'C': compute temporal entanglement entropy from correlation approach, 'L': compute Lohschmidt echo
write_mode = sys.argv[2] #if the argument is 'w', overwrite file if it exists, otherwise append if it exists
# lattice sites (in the environment):
nsites = int(sys.argv[3]) #total number of sites, including the impurity
# model parameters:
del_t = float(sys.argv[4])
Jx = float(sys.argv[5])  #* np.pi/2 #0.5# 0.31 # 0.31
Jy =float(sys.argv[6])# * np.pi/2#np.pi/4+0.3#np.pi/4
g =float(sys.argv[7]) #* np.pi/2#np.pi/4+0.3
init_state = int(sys.argv[8])#0: thermal e^{-\beta XX}, 1: Bell pairs, 2: BCS_GS, 3: Inf. Temp.. Invalied entries will be set to Inf. Temp. (=3)
beta = float(sys.argv[9])#0.4  # temperature
mu_initial_state_left = float(sys.argv[10])
mu_initial_state_right = float(sys.argv[11])

Jx_coupling_right = Jx *0.3162
Jy_coupling_right = Jy *0.3162
Jx_coupling_left = Jx *0.3162
Jy_coupling_left = Jy *0.3162

Jp = (Jx + Jy)
Jm = (Jy - Jx)

Jp_coupling_right = (Jx_coupling_right + Jy_coupling_right)
Jm_coupling_right = (Jy_coupling_right - Jx_coupling_right)
Jp_coupling_left = (Jx_coupling_left + Jy_coupling_left)
Jm_coupling_left = (Jy_coupling_left - Jx_coupling_left)

print('mode', mode)
print('write_mode', write_mode)
print('nsites', nsites)
print('del_t', del_t)
print('Jx', Jx)
print('Jy', Jy)
print('g', g)
print('beta', beta)
print('init_state', init_state)
print('mu_init_state_left', mu_initial_state_left)
print('mu_init_state_right', mu_initial_state_right)
if mode == 'L':
    print('g_boundary_mag', g_boundary_mag)


iterator = 0 # iterator at beginning of job. If file exist and job is continued, this will be set to the appropraite value below
# total_time = 0 means one floquet-layer

# must be initialized as matrix that diagonalizes dressed density matrix
N_left = np.identity((nsites//2) *2, dtype=np.complex_)
N_right = np.identity((nsites//2) *2, dtype=np.complex_)

nsites_left = nsites // 2
nsites_right =nsites // 2

# find generators and matrices which diagonalize the composite Floquet operator:


G_XY_even_left, G_XY_odd_left, G_g_left, G_1_left = compute_generators(nsites_left, Jx, Jy, g, 0.)
G_XY_even_right, G_XY_odd_right, G_g_right, G_1_right = compute_generators(nsites_right,Jx, Jy, g,0.)


H_left = (G_XY_odd_left + G_XY_even_left) / 2
H_right = (G_XY_odd_right + G_XY_even_right) / 2

#print(H_left)

eigenvals_dressed_left, eigenvecs_dressed1 = linalg.eigh(H_left[:nsites_left,:nsites_left])
eigenvals_dressed_right, eigenvecs_dressed2 = linalg.eigh(H_right[:nsites_right,:nsites_right])
print(eigenvals_dressed_left)

eigenvals_dressed_left=np.concatenate((eigenvals_dressed_left,-eigenvals_dressed_left)) 
eigenvals_dressed_right=np.concatenate((eigenvals_dressed_right,-eigenvals_dressed_right)) 


N_left = np.bmat([[eigenvecs_dressed1,np.zeros((nsites_left,nsites_left))],[np.zeros((nsites_left,nsites_left)), eigenvecs_dressed1.conj()]])
N_right = np.bmat([[eigenvecs_dressed2,np.zeros((nsites_right,nsites_right))],[np.zeros((nsites_right,nsites_right)), eigenvecs_dressed2.conj()]])

n_k_left = np.zeros((2*nsites_left))
n_k_left.fill(.5)
n_k_right = np.zeros((2*nsites_right)) 
n_k_right.fill(.5)
#print(N_right)



Lambda_right = np.diag(n_k_right)
potentials = [0.0,0.01,.02,0.04,.08,.16]#,0.005,0.01,0.015,0.02,0.025,0.03,0.035,0.04]#,0.045,0.05,0.055,0.06,0.07,0.08,0.1,0.2,0.3,0.5]
#potentials = [.9,1.,1.1,1.2]
for mu in potentials:
    mu_initial_state_left = -mu/2 
    mu_initial_state_right = mu/2 
    #print(mu_initial_state_left, mu_initial_state_right, beta)
    if init_state == 2:
       
        print('mu',mu)
        for k in range (nsites_left):
            n_k_left[k] = np.heaviside(2 * eigenvals_dressed_left[k]   - mu_initial_state_left , 0.)
            n_k_left[k + nsites_left] = 1. -  np.heaviside(2 * eigenvals_dressed_left[k]   - mu_initial_state_left , 0.)

            #n_k_left[k] =  1./(1. + np.exp(-beta * (eigenvals_dressed_left[k]   - mu_initial_state_left  ))) 
            #n_k_left[nsites_left+k] = 1./(1. + np.exp( beta * (eigenvals_dressed_left[k]  - mu_initial_state_left ))) 
       
        for k in range (nsites_right):
            n_k_right[k] = np.heaviside(2 * eigenvals_dressed_right[k]   - mu_initial_state_right , 0.)
            n_k_right[k + nsites_right] = 1. -  np.heaviside(2 * eigenvals_dressed_right[k]   - mu_initial_state_right , 0.)
            #n_k_right[k] = 1./(1. + np.exp(-beta * (eigenvals_dressed_right[k] - mu_initial_state_right ))) 
            #n_k_right[nsites_right+k] =1./(1. + np.exp(beta * (eigenvals_dressed_right[k]  - mu_initial_state_right )))

        Lambda_left = np.einsum('ij,j,jk->ik',N_left, n_k_left, N_left.T.conj(),optimize=True)
        Lambda_right = np.einsum('ij,j,jk->ik',N_right, n_k_right, N_right.T.conj(),optimize=True)



    Lambda_imp = np.empty(2) 
    Lambda_imp.fill(.5)


    Lambda = block_diag(Lambda_left[:nsites_left,:nsites_left], 0. ,Lambda_right[:nsites_right,:nsites_right], Lambda_left[nsites_left:,nsites_left:], 1. ,Lambda_right[nsites_right:,nsites_right:])
    Lambda[nsites:nsites+nsites_left,:nsites_left] = Lambda_left[nsites_left:,:nsites_left]
    Lambda[:nsites_left,nsites:nsites + nsites_left] = Lambda_left[:nsites_left,nsites_left:]
    Lambda[nsites+nsites_left+1:,nsites_left+1:nsites] = Lambda_right[nsites_right:,:nsites_right]
    Lambda[nsites_left+1:nsites,nsites+nsites_left+1:] = Lambda_right[:nsites_right,nsites_right:]

    np.set_printoptions(linewidth=np.nan, precision=6, suppress=True)
    #print(Lambda)
    #print(Lambda.shape)
    #single-species evolution
    G_XY_even, G_XY_odd, G_g, G_1 = compute_generators(nsites, Jx, Jy, g, 0.)
    #adapt coupling to impurity
    G_XY_even[nsites//2, nsites//2 + 1] =  +Jp_coupling_right
    G_XY_even[nsites//2 + 1, nsites//2] =  +Jp_coupling_right
    G_XY_even[nsites//2, nsites//2 + nsites + 1] =  - Jm_coupling_right
    G_XY_even[nsites//2 + 1, nsites//2 + nsites] = + Jm_coupling_right
    G_XY_even[nsites//2 + nsites, nsites//2 + 1] = + Jm_coupling_right
    G_XY_even[nsites//2 + nsites + 1, nsites//2] = - Jm_coupling_right
    G_XY_even[nsites//2 + nsites, nsites//2 + nsites + 1] = - Jp_coupling_right
    G_XY_even[nsites//2 + nsites + 1, nsites//2 + nsites] = - Jp_coupling_right

    G_XY_odd[(nsites//2-1), (nsites//2-1) + 1] = + Jp_coupling_left
    G_XY_odd[(nsites//2-1) + 1, (nsites//2-1)] = + Jp_coupling_left
    G_XY_odd[(nsites//2-1), (nsites//2-1) + nsites + 1] = - Jm_coupling_left
    G_XY_odd[(nsites//2-1) + 1, (nsites//2-1) + nsites] = + Jm_coupling_left
    G_XY_odd[(nsites//2-1) + nsites, (nsites//2-1) + 1] = + Jm_coupling_left
    G_XY_odd[(nsites//2-1) + nsites + 1, (nsites//2-1)] = - Jm_coupling_left
    G_XY_odd[(nsites//2-1) + nsites, (nsites//2-1) + nsites + 1] = - Jp_coupling_left
    G_XY_odd[(nsites//2-1) + nsites + 1, (nsites//2-1) + nsites] = - Jp_coupling_left

    evol = expm(1.j * del_t * (G_XY_odd+G_XY_even)) 
    evol_even = expm(1.j*G_XY_even)
    evol_odd = expm(1.j*G_XY_odd)
  

    Lambda_tot = block_diag(Lambda,Lambda)# for two spin species
    evol_tot = block_diag(evol,evol)
    evol_tot_even = block_diag(evol_even,evol_even)
    evol_tot_odd = block_diag(evol_odd,evol_odd)


    filename = '/Users/julianthoenniss/Documents/PhD/data/' + str(nsites)+'_newstest2'

    with h5py.File(filename+'_spinfulpropag_mu{}'.format(mu) + ".hdf5", 'w') as f:
            dset_propag_IM = f.create_dataset('propag_exact', (len(time_array),),dtype=np.complex_)
            dset_left_curr = f.create_dataset('current_left_exact', (len(time_array),),dtype=np.complex_)
            dset_right_curr = f.create_dataset('current_right_exact', (len(time_array),),dtype=np.complex_)
            dset_n_even = f.create_dataset('n_even', (len(time_array),),dtype=np.complex_)
            dset_n_odd = f.create_dataset('n_odd', (len(time_array),),dtype=np.complex_)
            dset_nimp_tot = f.create_dataset('n_imp_tot', (len(time_array),),dtype=np.complex_)
            dset_nimp_even = f.create_dataset('n_imp_even', (len(time_array),),dtype=np.complex_)
            dset_curr_dirr = f.create_dataset('curr_dirr', (len(time_array),),dtype=np.complex_)
            dset_curr_cont = f.create_dataset('curr_cont', (len(time_array),),dtype=np.complex_)
            dset_nimp_odd = f.create_dataset('n_imp_odd', (len(time_array),),dtype=np.complex_)
            dset_n_expec = f.create_dataset('n_expec', (len(time_array),nsites),dtype=np.complex_)
            dset_times = f.create_dataset('times', (len(time_array),),dtype=np.float_)

    iter = -1
    print('init_Lambda')
    #print(Lambda_tot[:2*nsites,:2*nsites])
    for i in time_array:
        
        iter += 1
        t_2=i
        t_1=0
        #print(matrix_power(evol_tot,t_2)[:2*nsites,:2*nsites])
        Lambda_tot_temp = (matrix_power(evol_tot,t_2) @ Lambda_tot @ matrix_power(evol_tot.T.conj(),t_2))
        print('time: ', i,'iter',iter,' mu = ', mu_initial_state_left,mu_initial_state_right)

        Lambda_tot_temp_even = evol_tot_even @ Lambda_tot_temp @ evol_tot_even.T.conj()
        Lambda_tot_temp_odd = evol_tot_odd @ Lambda_tot_temp_even @ evol_tot_odd.T.conj()
        n_temp_tot_R = np.diag(Lambda_tot_temp[1*nsites +nsites_left: 2*nsites, 1*nsites +nsites_left: 2*nsites] + Lambda_tot_temp[3*nsites +nsites_left: 4*nsites,3*nsites +nsites_left: 4*nsites])
        n_temp_even_R = np.diag(Lambda_tot_temp_even[1*nsites +nsites_left: 2*nsites, 1*nsites +nsites_left: 2*nsites] + Lambda_tot_temp_even[3*nsites +nsites_left: 4*nsites,3*nsites +nsites_left: 4*nsites])
        n_temp_even_L = np.diag(Lambda_tot_temp_even[1*nsites : 1*nsites + nsites_left+1, 1*nsites : 1*nsites + nsites_left+1] + Lambda_tot_temp_even[3*nsites : 3*nsites + nsites_left+1, 3*nsites : 3*nsites + nsites_left+1])
        n_temp_odd_L =  np.diag(Lambda_tot_temp_odd[1*nsites : 1*nsites + nsites_left+1, 1*nsites : 1*nsites + nsites_left+1] + Lambda_tot_temp_odd[3*nsites : 3*nsites + nsites_left+1, 3*nsites : 3*nsites + nsites_left+1])
        with h5py.File(filename+'_spinfulpropag_mu{}'.format(mu)+".hdf5", 'a') as f:
            curr_data_L = f['current_left_exact']
            curr_data_R = f['current_right_exact']
            n_imp_tot = f['n_imp_tot']
            n_imp_even = f['n_imp_even']
            n_imp_odd = f['n_imp_odd']
            curr_dirr = f['curr_dirr']
            curr_cont = f['curr_cont']

            n_imp_tot [iter] = Lambda_tot_temp[nsites + nsites_left, nsites + nsites_left] #+ Lambda_tot_temp[3*nsites + nsites_left, 3*nsites + nsites_left]
            n_imp_even [iter] = Lambda_tot_temp_even[nsites + nsites_left, nsites + nsites_left] #+ Lambda_tot_temp_even[3*nsites + nsites_left, 3*nsites + nsites_left]
            n_imp_odd [iter] = Lambda_tot_temp_odd[nsites + nsites_left, nsites + nsites_left] #+ Lambda_tot_temp_odd[3*nsites + nsites_left, 3*nsites + nsites_left]
            curr_data_R[iter] = (np.trace(Lambda_tot_temp_even[nsites + nsites_left +1 :,nsites + nsites_left +1 :]) - np.trace(Lambda_tot_temp[nsites + nsites_left +1 :,nsites + nsites_left +1 :])) / del_t
            curr_data_L[iter] = (np.trace(Lambda_tot_temp_odd[nsites :nsites+nsites_left,nsites :nsites+nsites_left]) - np.trace(Lambda_tot_temp_even[nsites :nsites+nsites_left,nsites :nsites+nsites_left])) / del_t
    
            curr_dirr[iter] = -1.j *0.3162* (Lambda_tot_temp[nsites + nsites_left + 1, nsites + nsites_left] - Lambda_tot_temp[nsites + nsites_left , nsites + nsites_left + 1] ) 
            curr_cont[iter] = -1.j *0.3162* (Lambda_tot_temp[nsites + nsites_left +1 , nsites + nsites_left] - Lambda_tot_temp[nsites + nsites_left , nsites + nsites_left + 1] ) 
            
            n_expec = f['n_expec']
            n_expec[iter,:] = np.diag(Lambda_tot_temp[nsites:2*nsites,nsites:2*nsites]) + np.diag(Lambda_tot_temp[3*nsites:4*nsites,3*nsites:4*nsites])

            times = f['times']
            times[iter] = i 
            print(i,'current:',curr_cont[iter],times[iter])
