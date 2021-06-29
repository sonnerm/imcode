import numpy as np
from numpy import version
from numpy.lib.type_check import imag
from scipy.linalg import expm, schur, eigvals
from scipy import linalg
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
np.set_printoptions(suppress=False, linewidth=np.nan)

from matrix_diag import matrix_diag
from reorder_eigenvecs import reorder_eigenvecs
from correlator import correlator
from create_correlation_block import create_correlation_block
from entropy import entropy



def zero_to_nan(values):
    """Replace every 0 with 'nan' and return a copy."""
    return [float('nan') if x == 0 else x for x in values]


np.set_printoptions(precision=6, suppress=True)

max_time1 = 10
max_time2 = 20
stepsize1 = 8
stepsize2 = 16
entropy_values = np.zeros(
    (int(max_time1/stepsize1) + int(max_time2/stepsize2) + 3, max_time2 + stepsize2))
times = np.zeros(int(max_time1/stepsize1) + int(max_time2/stepsize2))
print 'here', int(max_time1/stepsize1), int(max_time2/stepsize2)
nsites = 4


Jx = 0.3
Jy = 1.06
g = 0#np.pi/4
beta = 0
iterator = 1

fig, ax = plt.subplots(2)
M_fw, M_fw_inverse, M_bw, M_bw_inverse,  eigenvalues_G_eff_fw, eigenvalues_G_eff_bw, nsites, f= matrix_diag(nsites, Jx, Jy, g)

T_xy = 1 / (1 + f * np.tan(Jx) * np.tan(Jy))
print 'T_xy', T_xy, 'f=', f


for time in range(stepsize1, max_time1, stepsize1):  # 90, nsites = 200,
    correlation_block = create_correlation_block(
        M_fw, M_fw_inverse, M_bw, M_bw_inverse,  eigenvalues_G_eff_fw, eigenvalues_G_eff_bw, nsites, time, Jx, Jy, g, beta, T_xy, f)
    time_cuts = np.arange(1, time)
    #times[iterator] = time
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

max_entropies = np.zeros(iterator)
half_entropies = np.zeros(iterator)
for i in range(iterator):
    max_entropies[i] = max(entropy_values[i, 1:])
    if entropy_values[i, 0] % 2 == 0:
        halftime = entropy_values[i, 0] / 2
        half_entropies[i] = entropy_values[i, int(halftime)]

print(max_entropies)
print(half_entropies)


ax[0].plot(entropy_values[:iterator, 0], max_entropies,
           'ro-', label=r'$max_t S$, ' + r'$J_x={},J_y={}, g={}, \beta = {}, L={}$'.format(Jx, Jy, g, beta, nsites))
ax[0].plot(entropy_values[:iterator, 0], zero_to_nan(half_entropies),
           'ro--', label=r'$S(t/2)$, ' + r'$J_x={},J_y={}, g={}, \beta = {}, L={}$'.format(Jx, Jy, g, beta, nsites), color='green')
ax[0].set_xlabel(r'$t$')

ax[0].yaxis.set_ticks_position('both')
ax[0].tick_params(axis="y", direction="in")
ax[0].tick_params(axis="x", direction="in")
ax[0].legend(loc="lower right")
# ax[0].set_ylim([0,1])
ax[0].set_xlabel(r'$t$')


gamma_test_range = 100
gamma_test_vals = np.zeros(gamma_test_range)
for i in range(gamma_test_range):
    gamma_test = 0
    for k in range(nsites):
        gamma_test += abs(M_fw[0, k] - M_fw[nsites, k])**2 * \
            np.cos(eigenvalues_G_eff_fw[k] * i)
    gamma_test_vals[i] = gamma_test
    # print 'i=',i, abs(M[0,i] - M[nsites,i])**2 , eigenvalues_G_eff[i], '\n'
    # print 'gamma_test', i, gamma_test

#ax[1].plot(np.arange(0,2 * nsites), np.sort(eigenvalues_G_eff),'.')

ax[1].plot(np.arange(0, gamma_test_range), gamma_test_vals,
           'ro-', label=r'$\gamma(t)$')
#ax[1].plot(np.arange(0,gamma_test_range), 5 * np.arange(0,gamma_test_range, dtype=float)**(-1.5), label= r'$t^{-3/2}$')
print('gamma', gamma_test_vals[0])
ax[1].set_xlabel(r'$t$')
# ax[1].set_xscale("log")
# ax[1].set_yscale("log")
# ax[1].set_ylim([1e-6,1])
ax[1].legend(loc="lower left")

ax[1].yaxis.set_ticks_position('both')
ax[1].tick_params(axis="y", direction="in")
ax[1].tick_params(axis="x", direction="in")

np.savetxt('../../../../data/correlation_approach/ent_entropy_Jx=' + str(Jx) + '_Jy=' + str(Jy) + '_g=' + str(g) + '_beta=' +
           str(beta) + '_L=' + str(nsites) + '.txt', entropy_values,  delimiter=' ', fmt='%1.5f')


plt.savefig('../../../../data/correlation_approach/ent_entropy_Jx=' + str(Jx) + '_Jy=' + str(Jy) + '_g=' +
            str(g) + '_beta=' + str(beta) + '_L=' + str(nsites) + '.png')
