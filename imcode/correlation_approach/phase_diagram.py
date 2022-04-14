
import numpy as np
from quimb import eigvals
from scipy.linalg import expm, logm
from compute_generators import compute_generators
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import linalg

nsites = 101

range_Jx = 20
range_Jy = 20


phase_diag = np.zeros((range_Jx,range_Jy)) #0: no edge mode, 1:pi-mode, 2:zero-mode, 3: both modes

for i in range (range_Jx):
    for j in range (range_Jy):
        g= 0
        Jx = np.pi/2 * (i+1) / (range_Jx + 1)
        Jy = np.pi/2 * (j+1) / (range_Jy + 1)
        G_XY_even, G_XY_odd, G_g, G_1 = compute_generators(nsites, Jx, Jy, g, 0)
        F_E =  expm(1.j* G_g) @ expm(1.j*  G_XY_even) @ expm(1.j * G_XY_odd)
        eigenvalues, eigenvectors = linalg.eig(F_E)  
        eigvals_abs_im_arg = np.argsort(abs(np.imag(eigenvalues)))
        #print(np.imag(eigenvalues[eigvals_abs_im_arg[:]]))
        if abs(np.imag(eigenvalues[eigvals_abs_im_arg[0]]))<1e-6:
            if np.real(eigenvalues[eigvals_abs_im_arg[0]]) < 0:
                print('pi mode')
                phase_diag[range_Jy-1-j,i] += 1
            else:
                print('zero mode')
                phase_diag[range_Jy-1-j,i] += 2
        if abs(np.imag(eigenvalues[eigvals_abs_im_arg[2]]))<1e-6:
            """     if np.real(eigenvalues[eigvals_abs_im_arg[2]]) < 0:
                print('pi mode')
                phase_diag[range_Jy-1-j,i] += 1
            else:
                print('zero mode')"""
            phase_diag[range_Jy-1-j,i] = 3
            print('..actually both!')
        print('\n')
            
        """fig, ax = plt.subplots(1)
        for k in range (len(eigenvalues)):
            x = np.real(eigenvalues[k])
            y = np.imag(eigenvalues[k])
            ax.scatter(x, y, label="star", marker="*", color="green", s=30)
        plt.show()"""
print(phase_diag)
phase_diag[range_Jy-1,range_Jx-1] = 3
fig, ax = plt.subplots(1)
contour_plot = ax.imshow(phase_diag, extent=[0,1,0,1],cmap='RdYlBu')
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

fig.colorbar(contour_plot, cax=cax)
plt.show()