import numpy as np
from numpy import version
from numpy.lib.type_check import imag
from scipy.linalg import expm, schur, eigvals
from scipy import linalg
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from numpy.linalg import multi_dot
import scipy.optimize
from reorder_eigenvecs import reorder_eigenvecs
from compute_generators import compute_generators
from add_cmplx_random_antisym import add_cmplx_random_antisym
np.set_printoptions(suppress=False, linewidth=np.nan)
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
plt.rcParams.update({
  "text.usetex": True,
  "font.family": "Helvetica",
  "font.size" : 10
})

def alg_decay(x,a,b):
    return a*np.log(x)+b
def matrix_diag(nsites, G_XY_even, G_XY_odd, G_g, G_1, Jx=0, Jy=0, g=0):
    #print(G_XY_odd)
    #print(G_XY_even)
    #print(G_g)
     # unitary gate is exp-product of exponentiated generators
    # gate that describes evolution non disconnected environment. first dimension: 0 = forward branch, 1 = backward branch
    U_E = np.zeros((2 * nsites, 2 * nsites), dtype=np.complex_)
    # gate that governs time evolution on both branches. first dimension: 0 = forward branch, 1 = backward branch
    U_eff = np.zeros((2, 2 * nsites, 2 * nsites), dtype=np.complex_)


    evol = expm(1.j*G_g) @ expm(1.j * G_XY_even) @ expm(1.j * G_XY_odd)   

    U_E = evol#.T.conj()

    U_eff[0] = expm(-G_1) @ evol.T.conj()  # forward branch
    U_eff[1] = expm(G_1) @ evol.T.conj()   # backward branch

    #print( 'U_E[0] = expm(1j*G_g) @ U_E[0]')
    #print( U_E[0] )
    #print( U_E[1] )
    #print( 'U_E[1] = expm(1j*G_g) @ U_E[1]')
   
    # generator of environment (always unitary)
    G_eff_E = -1j * linalg.logm(U_E)
    #print('G_eff_E')
    #print(G_eff_E)
   
    # G_eff is equivalent to generator for composed map (in principle obtainable through Baker-Campbell-Hausdorff)
    G_eff = np.zeros((2, 2 * nsites, 2 * nsites), dtype=np.complex_)
    G_eff[0] = -1j * linalg.logm(U_eff[0])
    G_eff[1] = +1j * linalg.logm(U_eff[1])

    #print('G_eff_E')
    #print(G_eff[0])
    #print('G_eff_b')
    #print(G_eff[1]-G_eff[0])
    
    # add small random part to G_eff to lift degenaracies, such that numerical diagnoalization is more stable
    rand_magn = 1.e-7
    rand_A = (np.random.rand(nsites,nsites) + 1.j * np.random.rand(nsites,nsites)) * rand_magn
    rand_B = add_cmplx_random_antisym(np.zeros((nsites,nsites), dtype=np.complex_), rand_magn)
    rand_C = add_cmplx_random_antisym(np.zeros((nsites,nsites), dtype=np.complex_), rand_magn)
    random_part = np.bmat([[rand_A,rand_B], [rand_C, -rand_A.T]])

    
    #G_eff_E += random_part
    G_eff[0] += random_part
    G_eff[1] += random_part

    # compute eigensystem of G_eff. Set of eigenvectors "eigenvectors_G_eff_fw/bw" diagnonalizes G_eff_fw/bw
    # first dimension: foward branch (index 0) and backward branch (index 1)
    eigenvalues_G_eff = np.zeros((2, 2 * nsites), dtype=np.complex_)
    #eigenvalues_G_eff_bw = np.zeros(2 * nsites, dtype=np.complex_)
    # no need for seperate forward and backward branch since F_E is always unitary, time evolution is equivalent on both branches
    #eigenvalues_G_eff_E = np.zeros(2 * nsites, dtype=np.complex_)
    # first dimension: foward branch (index 0) and backward branch (index 1)
    eigenvectors_G_eff = np.zeros(
        (2, 2 * nsites, 2 * nsites), dtype=np.complex_)
    #eigenvectors_G_eff_bw = np.zeros((2 * nsites, 2 * nsites), dtype=np.complex_)
    eigenvectors_G_eff_E = np.zeros(
        (2 * nsites, 2 * nsites), dtype=np.complex_)

    if abs(Jy) < 1e-10 or abs(Jx) < 1e-10:
        # take superposition with hermitian conjugate to stabilize numerical diagonalization (works only in unitary case, e.g. Ising-type coupling)
        eigenvalues_G_eff[0], eigenvectors_G_eff[0] = linalg.eig(
            0.5 * (G_eff[0] + G_eff[0].conj().T))
        # take superposition with hermitian conjugate to stabilize numerical diagonalization (works only in unitary case, e.g. Ising-type coupling)
        eigenvalues_G_eff[1], eigenvectors_G_eff[1] = linalg.eig(
            0.5 * (G_eff[1] + G_eff[1].conj().T))
        # take superposition with hermitian conjugate to stabilize numerical diagonalization (works only in unitary case, e.g. Ising-type coupling)
        eigenvalues_G_eff_E, eigenvectors_G_eff_E = linalg.eigh(
            0.5 * (G_eff_E + G_eff_E.conj().T))
    
    else:
        eigenvalues_G_eff[0], eigenvectors_G_eff[0] = linalg.eig(G_eff[0])
        eigenvalues_G_eff[1], eigenvectors_G_eff[1] = linalg.eig(G_eff[1])
        #eigenvalues_G_eff_bw, eigenvectors_G_eff_bw = linalg.eig(G_eff_bw)
      
        eigenvalues_G_eff_E, eigenvectors_G_eff_E = linalg.eigh( G_eff_E + random_part + random_part.T.conj())
        #eigenvalues_G_eff_E, eigenvectors_G_eff_E = linalg.eig(U_E)
   
    # check if found eigenvectors indeed fulfill eigenvector equation (trivial check)
    eigenvector_check = 0
    #eigenvector_check_bw = 0
    eigenvector_check_E = 0
    for branch in range(0, 2):
        for i in range(nsites):
            eigenvector_check += linalg.norm(np.dot(G_eff[branch], eigenvectors_G_eff[branch, :, i]) - np.dot(
                eigenvalues_G_eff[branch, i], eigenvectors_G_eff[branch, :, i]))
        #eigenvector_check_bw += linalg.norm(np.dot(G_eff_bw, eigenvectors_G_eff_bw[:, i]) - np.dot(eigenvalues_G_eff_bw[i], eigenvectors_G_eff_bw[:, i]))
    for i in range(nsites):
        eigenvector_check_E += linalg.norm(np.dot(G_eff_E, eigenvectors_G_eff_E[:, i]) - np.dot(
            eigenvalues_G_eff_E[i], eigenvectors_G_eff_E[:, i]))
    print('eigenvector_check (f/b/E)',
          eigenvector_check, '/', eigenvector_check_E)

    #print('forward eigenvalues')
    #print(eigenvalues_G_eff[0])

    #print('backward eigenvalues')
    #print(eigenvalues_G_eff[1])

    #print('environment eigenvalues')
    #print(eigenvalues_G_eff_E)

    # sort eigenvectors such that first half are the ones with positive real part of eigenvalues and second half the corresponding negative ones
    argsort_fw = np.argsort(- np.real(eigenvalues_G_eff[0]))
    argsort_bw = np.argsort(- np.real(eigenvalues_G_eff[1]))
    #argsort_bw = np.argsort(- np.real(eigenvalues_G_eff_bw))
    argsort_E = np.argsort(- eigenvalues_G_eff_E)
    #argsort_E = np.argsort(1.j*np.log(eigenvalues_G_eff_E))

    #eigenvalues_G_eff[0] = eigenvalues_G_eff[0,argsort_fw[:]]
    #eigenvalues_G_eff[1] = eigenvalues_G_eff[1,argsort_bw[:]]
    #eigenvalues_G_eff_E = eigenvalues_G_eff_E[argsort_E[:]]
    #print(eigenvalues_G_eff[0])
    #print(eigenvalues_G_eff[1])
    #print(eigenvalues_G_eff_E)

    M = np.zeros((2, 2 * nsites, 2 * nsites), dtype=np.complex_)
    #M_bw = np.zeros((2 * nsites, 2 * nsites), dtype=np.complex_)
    #M_E = np.zeros((2 * nsites, 2 * nsites), dtype=np.complex_)
    for i in range(nsites):  # sort eigenvectors and eigenvalues such that the first half are the ones with positive real part, and the second half have negative real parts
        M[0, :, i] = eigenvectors_G_eff[0, :, argsort_fw[i]]
        M[0, :, 2 * nsites - 1 - i] = eigenvectors_G_eff[0, :, argsort_fw[i + nsites]]
        M[1, :, i] = eigenvectors_G_eff[1, :, argsort_bw[i]]
        M[1, :, 2 * nsites - 1 - i] = eigenvectors_G_eff[1, :, argsort_bw[i + nsites]]
        #M_E[:, i] = eigenvectors_G_eff_E[:, argsort_E[i]]
        #M_E[:, 2 * nsites - 1 - i] = eigenvectors_G_eff_E[:, argsort_E[i + nsites]]
    M_E =  np.bmat([[eigenvectors_G_eff_E[nsites:2*nsites,argsort_E[:nsites]].conj(), eigenvectors_G_eff_E[0:nsites,argsort_E[:nsites]]],[eigenvectors_G_eff_E[0:nsites,argsort_E[:nsites]].conj(),eigenvectors_G_eff_E[nsites:2*nsites,argsort_E[:nsites]]]]) 


    #M1_inv = linalg.inv(M[1])
    #M0_inv = linalg.inv(M[0])

    #M_w = np.bmat([[M1_inv[:,nsites:2*nsites],M1_inv[:,:nsites]]])
    #print(M_w.shape)
    #print(M_w - M1_inv.T.conj())

    """fig, ax = plt.subplots(2)
    plt.subplots_adjust(hspace=0.1)
 
    fig.set_size_inches(1.35,1.7) 
    x = np.arange(0,nsites,1)
    k_vals = np.arange(-nsites,0) * np.pi / nsites
    C_real = np.real(M_E[0, :nsites] - M_E[nsites,0:nsites] )
    C_imag = np.imag(M_E[0, :nsites] - M_E[nsites,0:nsites] )
    #ax[0].plot(k_vals[:],(C_real), 'o',label=r'$|Re(\mathcal{C}_k)|$',ms=.8)
    #ax[0].plot(k_vals[:], (C_imag),'o', label=r'$|Im(\mathcal{C}_k)|$',ms=.8)
    ax[0].plot(k_vals[:], np.sqrt(C_real ** 2 + C_imag ** 2 ), label=r'$|\mathcal{C}_k|$')
    # perform the fit
    p0 = (1,0) # start with values near those we expect
    theta = np.arange(-4,5)
    
    ax[0].set_xlim(-np.pi,0)
    ax[0].set_ylim(bottom=0)
    ax[0].tick_params(axis="x",direction="in")
    ax[0].set_ylabel(r'$|\mathcal{C}_k(0)|$')
    #params, cv = scipy.optimize.curve_fit(alg_decay, x[nsites-10:nsites-1], abs(C_imag[nsites-10:nsites-1]), p0)
    #a ,b= params
    #print(a,b)
    
    #ax[0].plot(k_vals[0:nsites], alg_decay(x[:nsites],a,b),'--',label= 'linear. fit: '+ r'${}\cdot x+{}$'.format(round(a,6),round(b,6)))
    ax[0].set_xticks([-np.pi,-np.pi/2,0])
    ax[0].set_xticklabels([])

    ax[0].axhline(y=0,xmin=0., xmax=1,  linestyle='--',linewidth=1.,color='black')
    #ax[0].axhline(y=Jx**2,xmin=0.7, xmax=1,  linestyle='-',linewidth=1.,color='green')
    print('heeeeeeere', C_real[nsites - 1])
    #ax[0].legend()"""
    

    #print ('M0', M[0]@M[0].T.conj())
    #M[0] = reorder_eigenvecs(M[0], nsites)
    #M_E = reorder_eigenvecs(M_E, nsites)
    #M[1] = reorder_eigenvecs(M[1], nsites)
    #print('M_forward')
    #print(M[0])  # matrix that diagonalizes G_eff_fw
    #print(M[1])
    #print('M_backward')
    #print(M[1])  # matrix that diagonalizes G_eff_bw
    #print('M_environment')
    #print(M_E)  # matrix that diagonalizes G_eff_E
    #print(M_E @ M_E.T.conj()) 
    
    M_inverse = np.zeros((2, 2 * nsites, 2 * nsites), dtype=np.complex_)
    # diagonalize G_eff with eigenvectors to check:
    M_inverse[0] = linalg.inv(M[0])
    M_inverse[1] = linalg.inv(M[1])
    #M_E_inverse = linalg.inv(M_E)

    D = np.zeros((2, 2 * nsites, 2 * nsites), dtype=np.complex_)
    for branch in range(2):
        D[branch] = M_inverse[branch] @ G_eff[branch] @ M[branch]
        #print(M[(branch+1)%2].T.conj() @ G_eff[branch] @ M_inverse[(branch+1)%2].T.conj())
        #print(branch, '\n',D[branch])
        # this is the diagonal matrix with eigenvalues of G_eff on the diagonal
        # this makes sure that the order of the eigenvalues corresponds to the order of the eigenvectors in the matrix M
        eigenvalues_G_eff[branch] = D[branch].diagonal()
 
    
    # this is the diagonal matrix with eigenvalues of G_eff on the diagonal
    # this makes sure that the order of the eigenvalues corresponds to the order of the eigenvectors in the matrix M
    eigenvalues_G_eff_E =np.concatenate((eigenvalues_G_eff_E[argsort_E[2*nsites-1:nsites-1:-1]],eigenvalues_G_eff_E[argsort_E[:nsites]]))
    #print(eigenvalues_G_eff_E)
    #D_E = M_E.T.conj() @ G_eff_E @ M_E
    #print(D_E)
    #print( linalg.inv(M_E)  - M_E.T.conj())
    #eigenvalues_G_eff_E = D_E.diagonal()
    #print(np.real(eigenvalues_G_eff_E))
    #eigenvalues_G_eff_E, M_E = linalg.eig(U_E)
    #print(np.real(-1.j*np.log(eigenvalues_G_eff_E)))
    """
    #ax[1].plot(k_vals,eigenvalues_G_eff_E[0:nsites],color='C0')
    #ax[1].plot(k_vals,eigenvalues_G_eff_E[nsites:2*nsites],color='C0')
    #ax[1].plot(k_vals,eigenvalues_G_eff[0,0:nsites],color='C1')
    #ax[1].plot(k_vals,eigenvalues_G_eff[0,nsites:2*nsites],color='C1')
    #ax[1].plot(k_vals,eigenvalues_G_eff[1,0:nsites],color='C2')
    #ax[1].plot(k_vals,eigenvalues_G_eff[1,nsites:2*nsites],color='C2')
    

    #ax[1].plot(k_vals,eigenvalues_G_eff_E[0:nsites],color='C0')
    #ax[1].plot(k_vals,eigenvalues_G_eff_E[nsites:2*nsites],color='C0')
    #ax[1].plot(k_vals,np.real(eigenvalues_G_eff[0,0:nsites]),color='C1', linestyle='--')
    #ax[1].plot(k_vals,np.real(eigenvalues_G_eff[0,nsites:2*nsites]),color='C1', linestyle='--')
    #ax[1].plot(k_vals,np.real(eigenvalues_G_eff[1,0:nsites]),color='C2', linestyle=':')
    #ax[1].plot(k_vals,np.real(eigenvalues_G_eff[1,nsites:2*nsites]),color='C2', linestyle=':')

    ax[1].plot(k_vals,np.imag(eigenvalues_G_eff[0,0:nsites]),color='green', linestyle='--')
    ax[1].plot(k_vals,np.imag(eigenvalues_G_eff[0,nsites:2*nsites]),color='green', linestyle='--')
    #ax[1].plot(k_vals,np.imag(eigenvalues_G_eff[1,0:nsites]),color='C2', linestyle=':')
    #ax[1].plot(k_vals,np.imag(eigenvalues_G_eff[1,nsites:2*nsites]),color='C2', linestyle=':')
    
    #ax[1].plot(k_vals[:30],0.002*np.array((k_vals-k_vals[0]))[:30]**2, linestyle=':')
    #ax[1].axhline(y=2*g+2*Jx,xmin=0., xmax=1,  linestyle='--',linewidth=1.,color='black')
    #ax[1].axhline(y=2*g-2*Jx,xmin=0., xmax=1,  linestyle='--',linewidth=1.,color='green')
    ax[1].set_xlim(-np.pi/10,0)
    ax[1].set_xticks([-np.pi,-np.pi/2,0])
    ax[0].tick_params(axis="x",direction="inout")
    ax[1].set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$'])
    #ax[1].set_ylim(-2.5,2.5)
    ax[1].set_xlabel(r'$k$')
    #ax[1].yaxis.tick_right()
    #ax[0].yaxis.tick_right()
    ax[1].set_ylabel(r'$\phi_k$')
    #plt.show()
    #plt.savefig('0.31_pi4.pdf', bbox_inches='tight')
    #plt.savefig('0.3_0.3.pdf', bbox_inches='tight')
    """
    #print('D_fw= ')
    #print('D_bw= ')
    #print(D[1])
    #print(D[0])
    #print('D_E= ')
    #print(D_E)
    """
    # check if diagonalization worked
    diag_check = 0
    #diag_check_bw = 0
    diag_check_E = 0
    for branch in range(2):
        for i in range(0, 2 * nsites):
            for j in range(i + 1, 2 * nsites):
                diag_check += abs(D[branch, i, j])
    for i in range(0, 2 * nsites):
        for j in range(i + 1, 2 * nsites):
            diag_check_E += abs(D_E[i, j])
    print('diag_checks (fw+bw/E)', diag_check, '/', diag_check_E)

    f = 0
    for k in range(nsites):
        f += abs(M_E[0, k])**2 - abs(M_E[nsites, k])**2 + \
            2j * imag(M_E[0, k]*M_E[nsites, k].conj())
    print ('f', f)"""
    #eigenvalues_G_eff = ews_sorted
    #print('eigenvalues of G_eff_fw: ')
    #print(eigenvalues_G_eff[0])

    #print('eigenvalues of G_eff_bw: ')
    #print(eigenvalues_G_eff[1])

    #print ('eigenvalues of G_eff_bw: ')
    # print(eigenvalues_G_eff_bw)

    #print('eigenvalues of G_eff_E: ')
    #print(eigenvalues_G_eff_E)

    print('Diagonalization of generators completed..')
    
    # return M_fw, M_fw_inverse, M_bw, M_bw_inverse,  eigenvalues_G_eff_fw, eigenvalues_G_eff_bw, f
    
    return M, M_E, eigenvalues_G_eff, eigenvalues_G_eff_E
