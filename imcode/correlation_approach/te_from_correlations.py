import numpy as np
from create_correlation_block import create_correlation_block
from entropy import entropy
from IM_exponent import IM_exponent
from add_cmplx_random_antisym import add_cmplx_random_antisym
from plot_entropy import plot_entropy
from matrix_diag import matrix_diag


np.set_printoptions(linewidth=np.nan,precision=10, suppress=True )

#define fixed parameters:
#step sizes for total times t
max_time1 = 40
max_time2 =41
stepsize1 = 4
stepsize2 = 16

#lattice sites:
nsites = 50

#model parameters:
Jx = 0
Jy = 1.06
g = np.pi/4
beta = 0 # temperature

#define initial density matrix and determine matrix which diagonalizes it:
rho_t = np.zeros((nsites,nsites), dtype=np.complex_)#this is the EXPONENT of the gaussian density matrix 
N_t = np.identity( 2 * nsites, dtype=np.complex_) # must be initialized as matrix that diagonalizes dressed density matrix

#initialize arrays in which entropy values and corresponding time-cuts are stored
entropy_values = np.zeros((int(max_time1/stepsize1) + int(max_time2/stepsize2) + 3, max_time2 + stepsize2))#entropies
times = np.zeros(int(max_time1/stepsize1) + int(max_time2/stepsize2))#time cuts

#find generators and matrices which diagonalize them:
M, M_inverse, eigenvalues_G_eff, f= matrix_diag(nsites, Jx, Jy, g)

iterator = 1
for time in range(stepsize1, max_time1, stepsize1):
    B, ising_gamma_times, gamma_test_vals = IM_exponent(M, M_inverse, eigenvalues_G_eff, f, N_t, nsites, time, Jx, Jy,g, rho_t)
    #B = add_cmplx_random_antisym(B, 1e-10)#add random antisymmetric part to matrix to lift degeneracies and stabilize numerics
    correlation_block = create_correlation_block(B, time)
    time_cuts = np.arange(1, time)

    entropy_values[iterator, 0] = time
    for cut in time_cuts:
        entropy_values[iterator, cut] = entropy(correlation_block, time, cut)
    iterator += 1

"""
for time in range(max_time1, max_time2 + stepsize2, stepsize2):  # 90, nsites = 200,
    correlation_block = create_correlation_block(
        M, eigenvalues_G_eff, nsites, time, Jx, Jy, g, beta, T_xy, f)
    time_cuts = np.arange(1, time)
    #times[iterator] = time
    entropy_values[iterator, 0] = time
    for cut in time_cuts:
        entropy_values[iterator, cut] = entropy(correlation_block, time, cut)
    iterator += 1
"""

print(entropy_values)

plot_entropy(entropy_values,ising_gamma_times, gamma_test_vals, iterator, Jx, Jy, g, beta, nsites)