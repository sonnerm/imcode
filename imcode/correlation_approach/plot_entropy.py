import matplotlib.pyplot as plt
import numpy as np
from utils import zero_to_nan
def plot_entropy(entropy_values,ising_gamma_times, gamma_test_vals, iterator, Jx, Jy, g, beta, nsites):
    #plot
    
    max_entropies = np.zeros(iterator)
    half_entropies = np.zeros(iterator)
    for i in range(iterator):
        max_entropies[i] = max(entropy_values[i, 1:])
        if entropy_values[i, 0] % 2 == 0:
            halftime = entropy_values[i, 0] / 2
            half_entropies[i] = entropy_values[i, int(halftime)]

    print(max_entropies)
    print(half_entropies)

    fig, ax = plt.subplots(2)
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


    ax[1].plot(ising_gamma_times, gamma_test_vals,
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


    mac_path = '/Users/julianthoenniss/Documents/Studium/PhD/data/correlation_approach'
    work_path = '/Users/julianthoenniss/Documents/PhD/data'
    fiteo1_path = '/home/thoennis/data/correlation_apporach/data/'
    np.savetxt(work_path + 'ent_entropy_Jx=' + str(Jx) + '_Jy=' + str(Jy) + '_g=' + str(g) + '_beta=' +
            str(beta) + '_L=' + str(nsites) + '.txt', entropy_values,  delimiter=' ', fmt='%1.5f')


    plt.savefig(work_path + 'ent_entropy_Jx=' + str(Jx) + '_Jy=' + str(Jy) + '_g=' +
                str(g) + '_beta=' + str(beta) + '_L=' + str(nsites) + '.png')
