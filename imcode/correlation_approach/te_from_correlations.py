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
max_time1 = 40
max_time2 = 60
stepsize1 = 10
stepsize2 = 18

# lattice sites:
nsites = 150

# model parameters:
Jx = np.pi/4  # 0.31
Jy = np.pi/4
g = 0  # 0.31#1.06#np.pi/4
beta = 0  # temperature
alpha_0_square = (np.cos(2 * Jx) + np.cos(2 * Jy)) / 2.
gamma_test_range = 100

# define initial density matrix and determine matrix which diagonalizes it:
# this is the EXPONENT of the BARE gaussian density matrix
rho_0 = np.zeros((2 * nsites, 2 * nsites), dtype=np.complex_)
# must be initialized as matrix that diagonalizes dressed density matrix
N_t = np.identity(2 * nsites, dtype=np.complex_)

# initialize arrays in which entropy values and corresponding time-cuts are stored
entropy_values = np.zeros((int(max_time1/stepsize1) +
                          int(max_time2/stepsize2) + 3, max_time2 + stepsize2))  # entropies
times = np.zeros(int(max_time1/stepsize1) +
                 int(max_time2/stepsize2))  # time cuts
ising_gamma_times = np.zeros(gamma_test_range)
ising_gamma_values = np.zeros(gamma_test_range)
# find generators and matrices which diagonalize them:
G_XY_even, G_XY_odd, G_g, G_1 = compute_generators(nsites, Jx, Jy, g)
evolution_matrix, F_E_prime, F_E_prime_dagger = evolution_matrix(
    nsites, G_XY_even, G_XY_odd, G_g, G_1, Jx, Jy, g)
#M, eigenvalues_G_eff, f= matrix_diag(nsites, G_XY_even, G_XY_odd, G_g, G_1, Jx, Jy, g)
#ising_gamma_times, ising_gamma_values = ising_gamma(M,eigenvalues_G_eff, nsites, gamma_test_range)

iterator = 1
# total_time = 0 means one floquet-layer
for total_time in range(2, max_time1, stepsize1):
    nbr_Floquet_layers = total_time + 1
    # dressed density matrix: (F_E^\prime )^{t+1} \rho_0 (F_E^\prime )^{\dagger,t+1}
    correlation_block = np.identity(8 * total_time, dtype=np.complex_)
    if (abs(alpha_0_square) > 1e-8):
        print('alpha_0_square', alpha_0_square)
        rho_eigvals, N_t,  = dress_density_matrix(
            rho_0, F_E_prime, F_E_prime_dagger, nbr_Floquet_layers)
        B = IM_exponent(evolution_matrix, N_t, nsites,
                        nbr_Floquet_layers, Jx, Jy, rho_eigvals)
        # B = add_cmplx_random_antisym(B, 1e-10)#add random antisymmetric part to matrix to lift degeneracies and stabilize numerics
        correlation_block = create_correlation_block(B, nbr_Floquet_layers)

    time_cuts = np.arange(1, nbr_Floquet_layers)

    entropy_values[iterator, 0] = nbr_Floquet_layers
    for cut in time_cuts:
        entropy_values[iterator, cut] = entropy(
            correlation_block, nbr_Floquet_layers, cut)
    iterator += 1


for total_time in range(max_time1, max_time2 + stepsize2, stepsize2):  # 90, nsites = 200,
    nbr_Floquet_layers = total_time + 1
    # dressed density matrix: (F_E^\prime )^{t+1} \rho_0 (F_E^\prime )^{\dagger,t+1}
    correlation_block = np.identity(8 * total_time, dtype=np.complex_)
    if (abs(alpha_0_square) > 1e-8):
        rho_eigvals, N_t,  = dress_density_matrix(
            rho_0, F_E_prime, F_E_prime_dagger, nbr_Floquet_layers)
        B = IM_exponent(evolution_matrix, N_t, nsites,
                        nbr_Floquet_layers, Jx, Jy, rho_eigvals)
        # B = add_cmplx_random_antisym(B, 1e-10)#add random antisymmetric part to matrix to lift degeneracies and stabilize numerics
        correlation_block = create_correlation_block(B, nbr_Floquet_layers)

    time_cuts = np.arange(1, nbr_Floquet_layers)

    entropy_values[iterator, 0] = nbr_Floquet_layers
    for cut in time_cuts:
        entropy_values[iterator, cut] = entropy(
            correlation_block, nbr_Floquet_layers, cut)
    iterator += 1


print(entropy_values)

plot_entropy(entropy_values, ising_gamma_times,
             ising_gamma_values, iterator, Jx, Jy, g, beta, nsites)
