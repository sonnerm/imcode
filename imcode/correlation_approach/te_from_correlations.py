from evolution_matrix import evolution_matrix
from compute_generators import compute_generators
import numpy as np
from create_correlation_block import create_correlation_block
from entropy import entropy
from IM_exponent import IM_exponent
from add_cmplx_random_antisym import add_cmplx_random_antisym
from plot_entropy import plot_entropy
from matrix_diag import matrix_diag
from dress_density_matrix import dress_density_matrix
from compute_generators import compute_generators
from ising_gamma import ising_gamma
import math

np.set_printoptions(linewidth=np.nan, precision=2, suppress=True)

# define fixed parameters:
# step sizes for total times t
max_time1 = 7
max_time2 = 8
stepsize1 = 1
stepsize2 = 10

# lattice sites:
nsites = 8

# model parameters:
Jx = 0#0.5# 0.31 # 0.31
Jy = 0.7
g = 0.2
beta = 0  # temperature
beta_tilde = np.arctanh(np.tan(Jx) * np.tan(Jy))
alpha_0_square = (np.cos(2 * Jx) + np.cos(2 * Jy)) / 2.
gamma_test_range = 6

# define initial density matrix and determine matrix which diagonalizes it:
# this is the EXPONENT of the BARE gaussian density matrix
rho_0_exponent = np.zeros((2 * nsites, 2 * nsites), dtype=np.complex_)
# must be initialized as matrix that diagonalizes dressed density matrix
N_t = np.identity(2 * nsites, dtype=np.complex_)




# initialize arrays in which entropy values and corresponding time-cuts are stored
entropy_values = np.zeros((int(max_time1/stepsize1) +int(max_time2/stepsize2) + 3, max_time2 + stepsize2))  # entropies
times = np.zeros(int(max_time1/stepsize1) + int(max_time2/stepsize2))  # time cuts
ising_gamma_times = np.zeros(gamma_test_range)
ising_gamma_values = np.zeros(gamma_test_range)


# find generators and matrices which diagonalize the composite Floquet operator:
G_XY_even, G_XY_odd, G_g, G_1 = compute_generators(nsites, Jx, Jy, g, beta_tilde)
evolution_matrix, F_E_prime, F_E_prime_dagger = evolution_matrix(nsites, G_XY_even, G_XY_odd, G_g, G_1)

#M, M_E, eigenvalues_G_eff, f= matrix_diag(nsites, G_XY_even, G_XY_odd, G_g, G_1, Jx, Jy, g)
#ising_gamma_times, ising_gamma_values = ising_gamma(M,eigenvalues_G_eff, nsites, gamma_test_range)

iterator = 1
# total_time = 0 means one floquet-layer
for total_time in range(1, max_time1, stepsize1):

    nbr_Floquet_layers = total_time + 1
    correlation_block = np.identity(8 * total_time, dtype=np.complex_)

    n_expect, N_t = dress_density_matrix(rho_0_exponent, F_E_prime, F_E_prime_dagger, nbr_Floquet_layers)
    
    B = IM_exponent(evolution_matrix, N_t, nsites,nbr_Floquet_layers, Jx, Jy, beta_tilde, n_expect)

    correlation_block = create_correlation_block(B, nbr_Floquet_layers)

    time_cuts = np.arange(1, nbr_Floquet_layers)

    entropy_values[iterator, 0] = nbr_Floquet_layers

    for cut in time_cuts:
        entropy_values[iterator, cut] = entropy(correlation_block, nbr_Floquet_layers, cut)
    iterator += 1


for total_time in range(max_time1, max_time2 + stepsize2, stepsize2):  # 90, nsites = 200,
    nbr_Floquet_layers = total_time + 1
    correlation_block = np.identity(8 * total_time, dtype=np.complex_)

    n_expect, N_t = dress_density_matrix(rho_0_exponent, F_E_prime, F_E_prime_dagger, nbr_Floquet_layers)
    
    B = IM_exponent(evolution_matrix, N_t, nsites,nbr_Floquet_layers, Jx, Jy, beta_tilde, n_expect)

    correlation_block = create_correlation_block(B, nbr_Floquet_layers)

    time_cuts = np.arange(1, nbr_Floquet_layers)

    entropy_values[iterator, 0] = nbr_Floquet_layers

    for cut in time_cuts:
        entropy_values[iterator, cut] = entropy(correlation_block, nbr_Floquet_layers, cut)
    iterator += 1

np.set_printoptions(linewidth=np.nan, precision=5, suppress=True)
print(entropy_values)

plot_entropy(entropy_values, iterator, Jx, Jy, g,  nsites, ising_gamma_times, ising_gamma_values)
