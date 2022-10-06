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
def cl(va,lt,cmap='Reds',invert=False,margin=0.1,lowcut=0,upcut=1): 
    cmap = plt.cm.get_cmap(cmap)
    ind=list(lt).index(va) 
    if len(lt)>1:
        rt=ind/(len(lt)-1)
        rt*=1-min(1-margin,lowcut+1-upcut)
        rt+=lowcut
    else:
        rt=0
        rt+=lowcut
    rt=rt*(1-2*margin)+margin
    if invert:
        rt=1-rt    
    return cmap(rt)


np.set_printoptions(linewidth=np.nan, precision=1, suppress=True)

# define fixed parameters:
# step sizes for total times t
time_0 = 2
max_time1 = 49
max_time2 = 59
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
mu_initial_state_left = float(sys.argv[10])
mu_initial_state_right = float(sys.argv[11])

Jx_coupling = Jx *0.5
Jy_coupling = Jy *0.5

Jp = (Jx + Jy)
Jm = (Jy - Jx)

Jp_coupling = (Jx_coupling + Jy_coupling)
Jm_coupling = (Jy_coupling - Jx_coupling)

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
print('max diff', np.amax(H_eff_left - (G_XY_odd_left + G_XY_even_left)))

eigenvals_dressed_left, eigenvecs_dressed1 = linalg.eigh(H_eff_left[:nsites_left,:nsites_left])
eigenvals_dressed_right, eigenvecs_dressed2 = linalg.eigh(H_eff_right[:nsites_right,:nsites_right])
np.set_printoptions(linewidth=np.nan, precision=7, suppress=True)
#print('H_eff_right',H_eff_right)

eigenvals_dressed_left=np.concatenate((eigenvals_dressed_left,-eigenvals_dressed_left))
eigenvals_dressed_right=np.concatenate((eigenvals_dressed_right,-eigenvals_dressed_right))


print('eigenvals_dressed_right',eigenvals_dressed_right)
#argsort_left = np.argsort(- eigenvals_dressed_left)
#argsort_right = np.argsort(- eigenvals_dressed_right)


print('egcalsleft',eigenvals_dressed_left)
"""
#By knowledge of the structure of N, i.e. (N = [[A, B^*],[B, A^*]]), we can construct the right part of the matrix from the left part of the matrix "eigenvecs_dressed", such that all phase factors and the order of eigenvectors are as desired.
N_left = np.identity(H_eff_left.shape[0])
if abs(np.max(eigenvals_dressed_left)) > 1.e-6:
    N_left =   np.bmat([[eigenvecs_dressed1[nsites_left:2*(nsites_left),argsort_left[:nsites_left]].conj(), eigenvecs_dressed1[0:nsites_left,argsort_left[:nsites_left]]],[eigenvecs_dressed1[0:nsites_left,argsort_left[:nsites_left]].conj(),eigenvecs_dressed1[nsites_left:2*(nsites_left),argsort_left[:nsites_left]]]]) 
N_right = np.identity(H_eff_right.shape[0])
if abs(np.max(eigenvals_dressed_right)) > 1.e-6:
    N_right =   np.bmat([[eigenvecs_dressed2[nsites_right:2*(nsites_right),argsort_right[:nsites_right]].conj(), eigenvecs_dressed2[0:nsites_right,argsort_right[:nsites_right]]],[eigenvecs_dressed2[0:nsites_right,argsort_right[:nsites_right]].conj(),eigenvecs_dressed2[nsites_right:2*(nsites_right),argsort_right[:nsites_right]]]]) 
"""
N_left = np.bmat([[eigenvecs_dressed1,np.zeros((nsites_left,nsites_left))],[np.zeros((nsites_left,nsites_left)), eigenvecs_dressed1.conj()]])
N_right = np.bmat([[eigenvecs_dressed2,np.zeros((nsites_right,nsites_right))],[np.zeros((nsites_right,nsites_right)), eigenvecs_dressed2.conj()]])
#print(N_left.T.conj() @ H_eff_left @ N_left)
n_k_left = np.zeros((2*nsites_left))
n_k_left.fill(.5)
n_k_right = np.zeros((2*nsites_right)) 
n_k_right.fill(.5)
#print(N_right)



Lambda_right = np.diag(n_k_right)
potentials = [.0]
for mu in potentials:
    mu_initial_state_left = -mu/2
    mu_initial_state_right = mu/2
    print(mu_initial_state_left, mu_initial_state_right, beta)
    if init_state == 2:
        print('creating finite temperature/voltage data')
        """for k in range (nsites_left):
            n_k_left[k] =  1./(1. + np.exp(-beta * (eigenvals_dressed_left[k]  - mu_initial_state_left  ))) 
            n_k_left[nsites_left+k] = 1./(1. + np.exp(beta * (eigenvals_dressed_left[k] - mu_initial_state_left ))) 
        
        print('occ',np.sum(n_k_left[nsites_left:]/nsites_left))
        for k in range (nsites_right):
            n_k_right[k] = 1./(1. + np.exp(-beta * (eigenvals_dressed_right[k] - mu_initial_state_right ))) 
            n_k_right[nsites_right+k] =1./(1. + np.exp(beta * (eigenvals_dressed_right[k] - mu_initial_state_right ))) """
        print('mu',mu)
        for k in range (nsites_left):
            n_k_left[k] =  1./(1. + np.exp(beta * (eigenvals_dressed_left[k] - eigenvals_dressed_left[k+nsites_left]  - mu_initial_state_left  ))) 
            n_k_left[nsites_left+k] = 1./(1. + np.exp( -beta * (eigenvals_dressed_left[k] - eigenvals_dressed_left[k+nsites_left] - mu_initial_state_left ))) 
       
        for k in range (nsites_right):
            n_k_right[k] = 1./(1. + np.exp(beta * (eigenvals_dressed_right[k] - eigenvals_dressed_right[k+nsites_right]- mu_initial_state_right ))) 
            n_k_right[nsites_right+k] =1./(1. + np.exp(- beta * (eigenvals_dressed_right[k] - eigenvals_dressed_right[k+nsites_right] - mu_initial_state_right )))

        Lambda_left = np.einsum('ij,j,jk->ik',N_left, n_k_left, N_left.T.conj(),optimize=True)
        Lambda_right = np.einsum('ij,j,jk->ik',N_right, n_k_right, N_right.T.conj(),optimize=True)
    #print('nk-right',n_k_right)
        #Lambda_left = Lambda_right
    #print('occ_r',np.trace(Lambda_left[nsites_left:,nsites_left:]/nsites_left))
    #print('occ_l',np.trace(Lambda_right[nsites_right:,nsites_right:]/nsites_right))
    Lambda_imp = np.empty(2) 
    Lambda_imp.fill(.5)


    Lambda = block_diag(Lambda_left[:nsites_left,:nsites_left], 0. ,Lambda_right[:nsites_right,:nsites_right], Lambda_left[nsites_left:,nsites_left:], 1. ,Lambda_right[nsites_right:,nsites_right:])
    Lambda[nsites:nsites+nsites_left,:nsites_left] = Lambda_left[nsites_left:,:nsites_left]
    Lambda[:nsites_left,nsites:nsites + nsites_left] = Lambda_left[:nsites_left,nsites_left:]
    Lambda[nsites+nsites_left+1:,nsites_left+1:nsites] = Lambda_right[nsites_right:,:nsites_right]
    Lambda[nsites_left+1:nsites,nsites+nsites_left+1:] = Lambda_right[:nsites_right,nsites_right:]


    #single-species evolution
    G_XY_even, G_XY_odd, G_g, G_1 = compute_generators(nsites, Jx, Jy, g, beta_tilde)
    #adapt coupling to impurity
    G_XY_even[nsites//2, nsites//2 + 1] =  +Jp_coupling
    G_XY_even[nsites//2 + 1, nsites//2] =  +Jp_coupling
    G_XY_even[nsites//2, nsites//2 + nsites + 1] =  - Jm_coupling
    G_XY_even[nsites//2 + 1, nsites//2 + nsites] = + Jm_coupling
    G_XY_even[nsites//2 + nsites, nsites//2 + 1] = + Jm_coupling
    G_XY_even[nsites//2 + nsites + 1, nsites//2] = - Jm_coupling
    G_XY_even[nsites//2 + nsites, nsites//2 + nsites + 1] = - Jp_coupling
    G_XY_even[nsites//2 + nsites + 1, nsites//2 + nsites] = - Jp_coupling

    G_XY_odd[(nsites//2-1), (nsites//2-1) + 1] = + Jp_coupling
    G_XY_odd[(nsites//2-1) + 1, (nsites//2-1)] = + Jp_coupling
    G_XY_odd[(nsites//2-1), (nsites//2-1) + nsites + 1] = - Jm_coupling
    G_XY_odd[(nsites//2-1) + 1, (nsites//2-1) + nsites] = + Jm_coupling
    G_XY_odd[(nsites//2-1) + nsites, (nsites//2-1) + 1] = + Jm_coupling
    G_XY_odd[(nsites//2-1) + nsites + 1, (nsites//2-1)] = - Jm_coupling
    G_XY_odd[(nsites//2-1) + nsites, (nsites//2-1) + nsites + 1] = - Jp_coupling
    G_XY_odd[(nsites//2-1) + nsites + 1, (nsites//2-1) + nsites] = - Jp_coupling

    evol = (expm(1.j*G_XY_odd) @ expm(1.j*G_XY_even))
    evol_even = expm(1.j*G_XY_even)
    evol_odd = expm(1.j*G_XY_odd)
    #hopping on impurity between two spin species
    t=0.0# -3 * 0.5 * del_t

    G_hop = np.zeros((4 * nsites,4 * nsites))
    G_hop[nsites_left, 2*nsites + nsites_left] = t
    G_hop[2*nsites + nsites_left,  nsites_left] = t
    G_hop[3*nsites + nsites_left, nsites + nsites_left] = -t
    G_hop[nsites + nsites_left, 3*nsites + nsites_left] = -t

    Lambda_tot = block_diag(Lambda,Lambda)
    evol_tot = block_diag(evol,evol)
    evol_tot_even = block_diag(evol_even,evol_even)
    evol_tot_odd = block_diag(evol_odd,evol_odd)


    evol_tot = expm(1.j * G_hop) @ evol_tot 

    filename = '/Users/julianthoenniss/Documents/PhD/data/test'

    with h5py.File(filename+'_spinfulpropag' + ".hdf5", 'w') as f:
            dset_propag_IM = f.create_dataset('propag_exact', (30,),dtype=np.complex_)
            dset_left_curr = f.create_dataset('current_left_exact', (30,),dtype=np.complex_)
            dset_right_curr = f.create_dataset('current_right_exact', (30,),dtype=np.complex_)
            dset_n_even = f.create_dataset('n_even', (30,),dtype=np.complex_)
            dset_n_odd = f.create_dataset('n_odd', (30,),dtype=np.complex_)
            dset_nimp_tot = f.create_dataset('n_imp_tot', (30,),dtype=np.complex_)
            dset_nimp_even = f.create_dataset('n_imp_even', (30,),dtype=np.complex_)
            dset_nimp_odd = f.create_dataset('n_imp_odd', (30,),dtype=np.complex_)
            dset_n_expec = f.create_dataset('n_expec', (30,nsites),dtype=np.complex_)


    print (evol.shape,Lambda_tot.shape)
    for i in range (0,1):
        t_2=i
        t_1=0
        value = (matrix_power(evol_tot,t_2) @ Lambda_tot @ matrix_power(evol_tot.T.conj(),t_1))[nsites_left,nsites_left]
        with h5py.File(filename+'_spinfulpropag' + ".hdf5", 'a') as f:
            propag_data = f['propag_exact']
            propag_data[i] = value

        print(i, value)

    for i in range (0,30):
        t_2=i
        t_1=0
        Lambda_tot_temp = (matrix_power(evol_tot,t_2) @ Lambda_tot @ matrix_power(evol_tot.T.conj(),t_2))
        Lambda_tot_temp_even = evol_tot_even @ Lambda_tot_temp @ evol_tot_even.T.conj()
        Lambda_tot_temp_odd = evol_tot_odd @ Lambda_tot_temp_even @ evol_tot_odd.T.conj()
        n_temp_tot_R = np.diag(Lambda_tot_temp[1*nsites +nsites_left: 2*nsites, 1*nsites +nsites_left: 2*nsites] + Lambda_tot_temp[3*nsites +nsites_left: 4*nsites,3*nsites +nsites_left: 4*nsites])
        n_temp_even_R = np.diag(Lambda_tot_temp_even[1*nsites +nsites_left: 2*nsites, 1*nsites +nsites_left: 2*nsites] + Lambda_tot_temp_even[3*nsites +nsites_left: 4*nsites,3*nsites +nsites_left: 4*nsites])
        n_temp_even_L = np.diag(Lambda_tot_temp_even[1*nsites : 1*nsites + nsites_left+1, 1*nsites : 1*nsites + nsites_left+1] + Lambda_tot_temp_even[3*nsites : 3*nsites + nsites_left+1, 3*nsites : 3*nsites + nsites_left+1])
        n_temp_odd_L =  np.diag(Lambda_tot_temp_odd[1*nsites : 1*nsites + nsites_left+1, 1*nsites : 1*nsites + nsites_left+1] + Lambda_tot_temp_odd[3*nsites : 3*nsites + nsites_left+1, 3*nsites : 3*nsites + nsites_left+1])
        with h5py.File(filename+'_spinfulpropag' + ".hdf5", 'a') as f:
            curr_data_L = f['current_left_exact']
            curr_data_R = f['current_right_exact']
            n_imp_tot = f['n_imp_tot']
            n_imp_even = f['n_imp_even']
            n_imp_odd = f['n_imp_odd']

            n_imp_tot [i] = Lambda_tot_temp[nsites + nsites_left, nsites + nsites_left] #+ Lambda_tot_temp[3*nsites + nsites_left, 3*nsites + nsites_left]
            n_imp_even [i] = Lambda_tot_temp_even[nsites + nsites_left, nsites + nsites_left] #+ Lambda_tot_temp_even[3*nsites + nsites_left, 3*nsites + nsites_left]
            n_imp_odd [i] = Lambda_tot_temp_odd[nsites + nsites_left, nsites + nsites_left] #+ Lambda_tot_temp_odd[3*nsites + nsites_left, 3*nsites + nsites_left]
            curr_data_R[i] = (np.trace(Lambda_tot_temp_even[nsites + nsites_left +1 :,nsites + nsites_left +1 :]) - np.trace(Lambda_tot_temp[nsites + nsites_left +1 :,nsites + nsites_left +1 :])) / del_t
            curr_data_L[i] = (np.trace(Lambda_tot_temp_odd[nsites :nsites+nsites_left,nsites :nsites+nsites_left]) - np.trace(Lambda_tot_temp_even[nsites :nsites+nsites_left,nsites :nsites+nsites_left])) / del_t
            #n_even_data = f['n_even']
            #n_odd_data = f['n_odd']
            #n_even_data[i] = n_temp_even 
            #n_odd_data[i] = n_temp_odd 

            n_expec = f['n_expec']
            n_expec[i,:] = np.diag(Lambda_tot_temp[nsites:2*nsites,nsites:2*nsites]) + np.diag(Lambda_tot_temp[3*nsites:4*nsites,3*nsites:4*nsites])

            #print('left current',i, curr_data_R[i])

    #with h5py.File(filename+'_spinfulpropag' + ".hdf5", 'r') as f:
        #curr_data_L_store = f['current_left_exact']
        #curr_data_R_store = f['current_right_exact']
        #n_even_data_store = f['n_even']
        #n_odd_data_store = f['n_odd']
        
        #curr_data_L_store[:30] = curr_data_L[:30]
        #curr_data_R_store[:30] = curr_data_R[:30]

        #plt.plot(-curr_data_L[:],label=r'$I_L,\, \Delta \mu ={}$'.format(mu_initial_state_right-mu_initial_state_left))
        #plt.plot(curr_data_R[:],label=r'$I_R,\, \Delta \mu ={}$'.format(mu_initial_state_right-mu_initial_state_left))
        #plt.plot(n_even_data[:],label='even')
        #plt.plot(n_odd_data[:],label='odd')
        
#plt.legend()
#plt.savefig('current_benchmark.pdf')
"""
with h5py.File(filename+'_spinfulpropag' + ".hdf5", 'r') as f:
    n_expec = f['n_expec']
    weights = np.arange(300)
    for i in range (0,300,10):
        plt.plot(n_expec[i,:],color = cl(i,weights))

    plt.show()"""