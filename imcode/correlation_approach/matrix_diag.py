import numpy as np
from numpy import version
from numpy.lib.type_check import imag
from scipy.linalg import expm, schur, eigvals
from scipy import linalg
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from numpy.linalg import multi_dot
from reorder_eigenvecs import reorder_eigenvecs
from compute_generators import compute_generators
#from numpy.core.einsumfunc import einsum
np.set_printoptions(suppress=False, linewidth=np.nan)


def matrix_diag(nsites, G_XY_even, G_XY_odd, G_g, G_1, Jx=0, Jy=0, g=0):

     # unitary gate is exp-product of exponentiated generators
    # gate that describes evolution non disconnected environment. first dimension: 0 = forward branch, 1 = backward branch
    U_E = np.zeros((2, 2 * nsites, 2 * nsites), dtype=np.complex_)

    U_1 =  np.zeros(( 2 * nsites, 2 * nsites), dtype=np.complex_)
    if (abs(abs(np.tan(Jx) * np.tan(Jy)) - 1) > 1e-6 ):
        print (abs(np.tan(Jx) * np.tan(Jy)))
        U_1 = expm(G_1)
    # gate that governs time evolution on both branches. first dimension: 0 = forward branch, 1 = backward branch
    U_eff = np.zeros((2, 2 * nsites, 2 * nsites), dtype=np.complex_)

    U_E[0] =  expm(1.j * G_XY_even) @ expm(1.j * G_XY_odd)
    U_E[1] =  expm(-1.j * G_XY_odd) @ expm(-1.j * G_XY_even)

    # onsite kicks (used in KIC):
    # this has an effect only when local onsite kicks (as in KIC) are nonzero
    U_E[0] = expm(1j*G_g) @ U_E[0]
    U_E[1] = U_E[1] @ expm(-1j*G_g)

    print( 'U_E[0] = expm(1j*G_g) @ U_E[0]')
    print( U_E[0] )
    print( 'U_E[1] = expm(1j*G_g) @ U_E[1]')
    print( U_E[1] )
   
    # generator of environment (always unitary)
    G_eff_E = -1j * linalg.logm(U_E[0])

    
    U_eff[0] = U_E[0] @ U_1
    U_eff[1] =  U_1 @ U_E[1]

    # G_eff is equivalent to generator for composed map (in principle obtainable through Baker-Campbell-Hausdorff)
    G_eff = np.zeros((2, 2 * nsites, 2 * nsites), dtype=np.complex_)
    G_eff[0] = -1j * linalg.logm(U_eff[0])
    G_eff[1] = +1j * linalg.logm(U_eff[1])
    
    # add small random part to G_eff to lift degenaracies, such that numerical diagnoalization is more stable
    random_part = np.random.rand(2 * nsites, 2 * nsites) * 1e-14
    # symmetrize random part
    for i in range(2*nsites):
        for j in range(i, 2*nsites):
            random_part[i, j] = random_part[j, i]
    G_eff_E += random_part
    #G_eff[0] += random_part
    #G_eff[1] += random_part

    # compute eigensystem of G_eff. Set of eigenvectors "eigenvectors_G_eff_fw/bw" diagnonalizes G_eff_fw/bw
    # first dimension: foward branch (index 0) and backward branch (index 1)
    eigenvalues_G_eff = np.zeros((2, 2 * nsites), dtype=np.complex_)
    #eigenvalues_G_eff_bw = np.zeros(2 * nsites, dtype=np.complex_)
    # no need for seperate forward and backward branch since F_E is always unitary, time evolution is equivalent on both branches
    eigenvalues_G_eff_E = np.zeros(2 * nsites, dtype=np.complex_)
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
        eigenvalues_G_eff_E, eigenvectors_G_eff_E = linalg.eig(
            0.5 * (G_eff_E + G_eff_E.conj().T))

    else:
        eigenvalues_G_eff[0], eigenvectors_G_eff[0] = linalg.eig(G_eff[0])
        eigenvalues_G_eff[1], eigenvectors_G_eff[1] = linalg.eig(G_eff[1])
        #eigenvalues_G_eff_bw, eigenvectors_G_eff_bw = linalg.eig(G_eff_bw)
        eigenvalues_G_eff_E, eigenvectors_G_eff_E = linalg.eig(G_eff_E)

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

    print('forward eigenvalues')
    print(eigenvalues_G_eff[0])

    print('backward eigenvalues')
    print(eigenvalues_G_eff[1])

    print('environment eigenvalues')
    print(eigenvalues_G_eff_E)

    # sort eigenvectors such that first half are the ones with positive real part of eigenvalues and second half the corresponding negative ones
    argsort_fw = np.argsort(- np.real(eigenvalues_G_eff[0]))
    argsort_bw = np.argsort(- np.real(eigenvalues_G_eff[1]))
    #argsort_bw = np.argsort(- np.real(eigenvalues_G_eff_bw))
    argsort_E = np.argsort(- np.real(eigenvalues_G_eff_E))
    M = np.zeros((2, 2 * nsites, 2 * nsites), dtype=np.complex_)
    #M_bw = np.zeros((2 * nsites, 2 * nsites), dtype=np.complex_)
    M_E = np.zeros((2 * nsites, 2 * nsites), dtype=np.complex_)
    for i in range(nsites):  # sort eigenvectors and eigenvalues such that the first half are the ones with positive real part, and the second half have negative real parts
        M[0, :, i] = eigenvectors_G_eff[0, :, argsort_fw[i]]
        M[0, :, 2 * nsites - 1 - i] = eigenvectors_G_eff[0, :, argsort_fw[i + nsites]]
        M[1, :, i] = eigenvectors_G_eff[1, :, argsort_bw[i]]
        M[1, :, 2 * nsites - 1 - i] = eigenvectors_G_eff[1, :, argsort_bw[i + nsites]]
        M_E[:, i] = eigenvectors_G_eff_E[:, argsort_E[i]]
        M_E[:, 2 * nsites - 1 - i] = eigenvectors_G_eff_E[:, argsort_E[i + nsites]]

    #M[0] = reorder_eigenvecs(M[0], nsites)
    #M_E = reorder_eigenvecs(M_E, nsites)
    #M[1] = reorder_eigenvecs(M[1], nsites)
    print('M_forward')
    print(M[0])  # matrix that diagonalizes G_eff_fw
    print('M_backward')
    print(M[1])  # matrix that diagonalizes G_eff_bw
    print('M_environment')
    print(M_E)  # matrix that diagonalizes G_eff_E

    
    M_inverse = np.zeros((2, 2 * nsites, 2 * nsites), dtype=np.complex_)
    # diagonalize G_eff with eigenvectors to check:
    M_inverse[0] = linalg.inv(M[0])
    M_inverse[1] = linalg.inv(M[1])
    M_E_inverse = linalg.inv(M_E)

    D = np.zeros((2, 2 * nsites, 2 * nsites), dtype=np.complex_)
    for branch in range(2):
        D[branch] = np.dot(M_inverse[branch], G_eff[branch])
        # this is the diagonal matrix with eigenvalues of G_eff on the diagonal
        D[branch] = np.dot(D[branch], M[branch])
        # this makes sure that the order of the eigenvalues corresponds to the order of the eigenvectors in the matrix M
        eigenvalues_G_eff[branch] = D[branch].diagonal()

    D_E = np.dot(M_E_inverse, G_eff_E)
    # this is the diagonal matrix with eigenvalues of G_eff on the diagonal
    D_E = np.dot(D_E, M_E)
    # this makes sure that the order of the eigenvalues corresponds to the order of the eigenvectors in the matrix M
    eigenvalues_G_eff_E = D_E.diagonal()

    print('D_fw= ')
    print(D[0])
    print('D_bw= ')
    print(D[1])
    print('D_E= ')
    print(D_E)

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
    print ('f', f)
    #eigenvalues_G_eff = ews_sorted
    print('eigenvalues of G_eff_fw: ')
    print(eigenvalues_G_eff[0])

    print('eigenvalues of G_eff_bw: ')
    print(eigenvalues_G_eff[1])

    #print ('eigenvalues of G_eff_bw: ')
    # print(eigenvalues_G_eff_bw)

    print('eigenvalues of G_eff_E: ')
    print(eigenvalues_G_eff_E)

    print('Diagonalization of generators completed..')
    
    # return M_fw, M_fw_inverse, M_bw, M_bw_inverse,  eigenvalues_G_eff_fw, eigenvalues_G_eff_bw, f
    
    return M, eigenvalues_G_eff, f
