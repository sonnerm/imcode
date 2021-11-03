from evolution_matrix import evolution_matrix
from compute_generators import compute_generators
import numpy as np
import h5py
from create_correlation_block import create_correlation_block
from entropy import entropy
from IM_exponent import IM_exponent
from add_cmplx_random_antisym import add_cmplx_random_antisym
from plot_entropy import plot_entropy
from matrix_diag import matrix_diag
from dress_density_matrix import dress_density_matrix
from compute_generators import compute_generators
from ising_gamma import ising_gamma
from gm_integral import gm_integral
import math

np.set_printoptions(linewidth=np.nan, precision=6, suppress=True)

# define fixed parameters:
# step sizes for total times t
max_time1 = 3
max_time2 = 3
stepsize1 = 1
stepsize2 = 1

# lattice sites (in the environment):
nsites = 8
# model parameters:
del_t = 1.0
Jx =0.3 * del_t #0.5# 0.31 # 0.31
Jy =0.5* del_t#np.pi/4+0.3#np.pi/4
g =0* del_t #np.pi/4+0.3
mu_initial_state = 0
beta = 0.4  # temperature

beta_tilde = np.arctanh(np.tan(Jx) * np.tan(Jy))
alpha_0_square = (np.cos(2 * Jx) + np.cos(2 * Jy)) / 2.
gamma_test_range = 6

# define initial density matrix and determine matrix which diagonalizes it:
# this is the EXPONENT of the BARE gaussian density matrix
rho_0_exponent = np.zeros((2 * nsites, 2 * nsites), dtype=np.complex_)
# must be initialized as matrix that diagonalizes dressed density matrix
N_t = np.identity(2 * nsites, dtype=np.complex_)


#set location for data storage
mac_path = '/Users/julianthoenniss/Documents/Studium/PhD/data/correlation_approach/'
work_path = '/Users/julianthoenniss/Documents/PhD/data/'
fiteo1_path = '/home/thoennis/data/correlation_approach/'

filename = work_path + 'FS_Jx=' + str(Jx/del_t) + '_Jy=' + str(Jy/del_t) + '_g=' + str(g/del_t) + 'mu=' + str(mu_initial_state) +'_del_t=' + str(del_t)+ '_beta=' + str(beta) + '_L=' + str(nsites) 



# initialize arrays in which entropy values and corresponding time-cuts are stored
#entropy_values = np.zeros((int(max_time1/stepsize1) +int(max_time2/stepsize2) + 3, max_time2 + stepsize2))  # entropies
#times = np.zeros(int(max_time1/stepsize1) + int(max_time2/stepsize2))  # time cuts
ising_gamma_times = np.zeros(gamma_test_range)
ising_gamma_values = np.zeros(gamma_test_range)

with h5py.File(filename + ".hdf5", 'w') as f:
    dset_temp_entr = f.create_dataset('temp_entr', (max_time1//stepsize1 + (max_time2- max_time1)//stepsize2 + 1, max_time2 + stepsize2),dtype=np.float_)
    dset_entangl_specrt = f.create_dataset('entangl_spectr', (int(max_time1/stepsize1) + (max_time2- max_time1)//stepsize2 + 1, max_time2 + stepsize2, 8*(max_time2 + stepsize2)),dtype=np.float_)
    dset_IM_exponent = f.create_dataset('IM_exponent', (int(max_time1/stepsize1) + (max_time2- max_time1)//stepsize2 + 1, 4 * (max_time2 + stepsize2), 4 * (max_time2 + stepsize2)),dtype=np.complex_)
    dset_edge_corr = f.create_dataset('edge_corr', (int(max_time1/stepsize1) + (max_time2- max_time1)//stepsize2 + 1, 2 * (4*(max_time2 + stepsize2) - 1), 2 * (4*(max_time2 + stepsize2) - 1) ),dtype=np.complex_)
    #dset = f.create_dataset('bulk_corr', (int(max_time1/stepsize1) + (max_time2- max_time1)//stepsize2 + 1, 2 * (4*(max_time2 + stepsize2) - 1) *nsites, 2 * (4*(max_time2 + stepsize2) - 1) *nsites),dtype=np.complex_)
    dset_init_BCS_state = f.create_dataset('init_BCS_state', (nsites, nsites),dtype=np.complex_)
# find generators and matrices which diagonalize the composite Floquet operator:
#G_XY_even, G_XY_odd, G_g, G_1 = compute_generators(nsites, Jx, Jy, g, beta_tilde)
#evolution_matrix, F_E_prime, F_E_prime_dagger = evolution_matrix(nsites, G_XY_even, G_XY_odd, G_g, G_1)

#M, M_E, eigenvalues_G_eff, f= matrix_diag(nsites, G_XY_even, G_XY_odd, G_g, G_1, Jx, Jy, g)
#ising_gamma_times, ising_gamma_values = ising_gamma(M,eigenvalues_G_eff, nsites, gamma_test_range)

iterator = 1
# total_time = 0 means one floquet-layer
for total_time in range(1, max_time1, stepsize1):

    nbr_Floquet_layers = total_time + 1
    correlation_block = np.identity(8 * total_time, dtype=np.complex_)

    #n_expect, N_t = dress_density_matrix(rho_0_exponent, F_E_prime, F_E_prime_dagger, nbr_Floquet_layers)
    
    #B = IM_exponent(evolution_matrix, N_t, nsites,nbr_Floquet_layers, Jx, Jy, beta_tilde, n_expect)
    N_sites_needed_for_entr = nsites#2*nbr_Floquet_layers 
    B = gm_integral(Jx,Jy,g,mu_initial_state, beta, N_sites_needed_for_entr,nbr_Floquet_layers, filename, iterator)
    correlation_block = create_correlation_block(B, nbr_Floquet_layers)
    time_cuts = np.arange(1, nbr_Floquet_layers)
    #entropy_values[iterator, 0] = nbr_Floquet_layers
    for cut in time_cuts:
        #entropy_values[iterator, cut] = entropy(correlation_block, nbr_Floquet_layers, cut)
        with h5py.File(filename + '.hdf5', 'a') as f:
            entr_data = f['temp_entr']
            entr_data[iterator,0] = nbr_Floquet_layers
            entr_data[iterator,cut] = float(entropy(correlation_block, nbr_Floquet_layers, cut, iterator, filename))
    iterator += 1

final_iter_one = iterator - 1
for total_time in range(final_iter_one + stepsize2, max_time2 + stepsize2, stepsize2):  # 90, nsites = 200,
#for total_time in range(1, max_time1, stepsize1):
    nbr_Floquet_layers = total_time + 1
    correlation_block = np.identity(8 * total_time, dtype=np.complex_)

    #n_expect, N_t = dress_density_matrix(rho_0_exponent, F_E_prime, F_E_prime_dagger, nbr_Floquet_layers)
    
    #B = IM_exponent(evolution_matrix, N_t, nsites,nbr_Floquet_layers, Jx, Jy, beta_tilde, n_expect)
    N_sites_needed_for_entr = nsites#2*nbr_Floquet_layers 
    B = gm_integral(Jx,Jy,g,mu_initial_state, beta,N_sites_needed_for_entr,nbr_Floquet_layers, filename, iterator)
    correlation_block = create_correlation_block(B, nbr_Floquet_layers)
    time_cuts = np.arange(1, nbr_Floquet_layers)
    #entropy_values[iterator, 0] = nbr_Floquet_layers

    for cut in time_cuts:
        #entropy_values[iterator, cut] = entropy(correlation_block, nbr_Floquet_layers, cut)
        with h5py.File(filename + '.hdf5', 'a') as f:
            entr_data = f['temp_entr']
            entr_data[iterator,0] = nbr_Floquet_layers
            entr_data[iterator,cut] = float(entropy(correlation_block, nbr_Floquet_layers, cut, iterator, filename))
    iterator += 1


with h5py.File(filename + '.hdf5', 'r') as f:
   entr_data = f['temp_entr']
   np.set_printoptions(linewidth=np.nan, precision=8, suppress=True)
   print(entr_data[:])

#plot_entropy(entropy_values, iterator, Jx/del_t, Jy/del_t, g/del_t, del_t, beta, nsites, filename, ising_gamma_times, ising_gamma_values)
