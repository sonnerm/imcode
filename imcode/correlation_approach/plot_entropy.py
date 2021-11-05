import matplotlib.pyplot as plt
import numpy as np
from utils import zero_to_nan


def plot_entropy(entropy_values, iterator, Jx, Jy, g, del_t,beta, nsites, filename, ising_gamma_times= [0], gamma_test_vals=[0]):
    # plot

    max_entropies = np.zeros(iterator)
    half_entropies = np.zeros(iterator)
    for i in range(iterator):
        max_entropies[i] = max(entropy_values[i, 1:])
        if entropy_values[i, 0] % 2 == 0:
            halftime = entropy_values[i, 0] / 2
            half_entropies[i] = entropy_values[i, int(halftime)]
        else:
            halftime = entropy_values[i, 0] // 2
            half_entropies[i] = entropy_values[i, int(halftime)]

    print(max_entropies)
    print(half_entropies)

    nbr_plots = 1
    if len(ising_gamma_times) > 1:
            nbr_plots += 1
    fig, ax = plt.subplots(nbr_plots)

    ax_main_plot = 0
    if nbr_plots > 1:
            ax_main_plot = ax[0]
    else: 
        ax_main_plot = ax
    ax_main_plot.plot(entropy_values[:iterator, 0], max_entropies,
            'ro-', label=r'$max_t S$, ' + r'$J_x={},J_y={}, g={},  L={}$'.format(Jx, Jy, g, nsites))
    ax_main_plot.plot(entropy_values[:iterator, 0], zero_to_nan(half_entropies),
            'ro--', label=r'$S(t/2)$, ' + r'$J_x={},J_y={}, g={},  L={}$'.format(Jx, Jy, g, nsites), color='green')
    ax_main_plot.set_xlabel(r'$t$')

    ax_main_plot.yaxis.set_ticks_position('both')
    ax_main_plot.tick_params(axis="y", direction="in")
    ax_main_plot.tick_params(axis="x", direction="in")
    ax_main_plot.legend(loc="lower right")
    # ax_main_plot.set_ylim([0,1])
    ax_main_plot.set_xlabel(r'$t$')


    if len(ising_gamma_times) > 1:
        ax[1].plot(ising_gamma_times, gamma_test_vals,
                'ro-', label=r'$\gamma(t)$')
        # ax[1].plot(np.arange(0,gamma_test_range), 5 * np.arange(0,gamma_test_range, dtype=float)**(-1.5), label= r'$t^{-3/2}$')
        print('gamma', gamma_test_vals[0])
        ax[1].set_xlabel(r'$t$')
        # ax[1].set_xscale("log")
        # ax[1].set_yscale("log")
        # ax[1].set_ylim([1e-6,1])
        ax[1].legend(loc="lower left")

        ax[1].yaxis.set_ticks_position('both')
        ax[1].tick_params(axis="y", direction="in")
        ax[1].tick_params(axis="x", direction="in")


    
    np.savetxt(filename + '.txt', entropy_values,  delimiter=' ', fmt='%1.5f')


    plt.savefig(filename +'.png')
