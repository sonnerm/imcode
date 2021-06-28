import numpy as np
from numpy import version
from numpy.lib.type_check import imag
from scipy.linalg import expm, schur, eigvals
from scipy import linalg
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
np.set_printoptions(suppress=False, linewidth=np.nan)

def matrix_diag(nsites, Jx=0, Jy=0, g=0):
    # define generators for unitary transformation

    # G_XY - two-site gates (XX + YY)
    G_XY_odd = np.zeros((2 * nsites, 2 * nsites))
    G_XY_even = np.zeros((2 * nsites, 2 * nsites))

    Jp = (Jx + Jy)
    Jm = (Jy - Jx)

    if abs(Jm) < 1e-10:
        Jm = 1e-10
    if abs(g) < 1e-10:
        g = 1e-10

    eps = 1e-6  # lift degeneracy
    G_XY_odd[0, nsites - 1] += eps
    G_XY_odd[nsites - 1, 0] += eps
    G_XY_odd[nsites, 2 * nsites - 1] += -eps
    G_XY_odd[2 * nsites - 1, nsites] += -eps

    G_XY_odd[nsites - 1, nsites] -= eps
    G_XY_odd[0, 2 * nsites - 1] += eps
    G_XY_odd[2 * nsites - 1, 0] += eps
    G_XY_odd[nsites, nsites - 1] -= eps

    for i in range(0, nsites - 1, 2):
        G_XY_even[i, i + 1] = Jp
        G_XY_even[i + 1, i] = Jp
        G_XY_even[i, i + nsites + 1] = -Jm
        G_XY_even[i + 1, i + nsites] = Jm
        G_XY_even[i + nsites, i + 1] = Jm
        G_XY_even[i + nsites + 1, i] = -Jm
        G_XY_even[i + nsites, i + nsites + 1] = -Jp
        G_XY_even[i + nsites + 1, i + nsites] = -Jp

    for i in range(1, nsites - 1, 2):
        G_XY_odd[i, i + 1] = Jp
        G_XY_odd[i + 1, i] = Jp
        G_XY_odd[i, i + nsites + 1] = -Jm
        G_XY_odd[i + 1, i + nsites] = Jm
        G_XY_odd[i + nsites, i + 1] = Jm
        G_XY_odd[i + nsites + 1, i] = -Jm
        G_XY_odd[i + nsites, i + nsites + 1] = - Jp
        G_XY_odd[i + nsites + 1, i + nsites] = - Jp

    # G_g - single body kicks
    G_g = np.zeros((2 * nsites, 2 * nsites))
    for i in range(nsites):
        G_g[i, i] = - 2 * g
        G_g[i + nsites, i + nsites] = 2 * g

    # G_1 - residual gate coming from projecting interaction gate of xy-model on the vacuum at site 0
    G_1 = np.zeros((2 * nsites, 2 * nsites))

    beta_tilde = np.arctanh(np.tan(Jx) * np.tan(Jy))

    G_1[0, 0] = 2 * beta_tilde
    G_1[nsites, nsites] = -2 * beta_tilde

    # give out explicit form of generators
    print('G_XY_even = ')
    print(G_XY_even)

    print('G_XY_odd = ')
    print(G_XY_odd)

    print('G_g = ')
    print(G_g)

    print('G_1 = ')
    print(G_1)

    # unitary gate is exp-product of exponentiated generators
    U_E_fw = np.zeros((2 * nsites, 2 * nsites), dtype=np.complex_)#gate that describes evolution non disconnected environment
    U_E_bw = np.zeros((2 * nsites, 2 * nsites), dtype=np.complex_)#gate that describes evolution non disconnected environment
    U_fw = np.zeros((2 * nsites, 2 * nsites), dtype=np.complex_)#gate that governs time evolution on forward branch
    U_bw = np.zeros((2 * nsites, 2 * nsites), dtype=np.complex_)#gate that governs time evolution on backward branch

    if abs(Jy) < 1e-10 or abs(Jx) < 1e-10:#for Ising-type couplings, the even and odd gates commute and can be added in the exponent (trivial Backer-Campbell Haussdorff)
        G_XY = G_XY_even + G_XY_odd
        U_E_fw = expm(1j * G_XY)
        U_E_bw = U_E_fw
    else:#for xy model, even and odd gates do not commute 
        U_E_fw = np.matmul(expm(1.j * G_XY_even), expm(1.j * G_XY_odd))
        U_E_bw = np.matmul(expm(-1.j * G_XY_odd), expm(-1.j * G_XY_even))


    #onsite kicks (used in KIC):
    U_fw = np.matmul(expm(1j*G_g), U_E_fw)#this has an effect only when local onsite kicks (as in KIC) are nonzero
    U_bw = np.matmul(U_E_bw, expm(-1j*G_g))

    #non-unitary gate stemming from vacuum projections (note that there is no imaginary j in from of G_1)
    U_fw = np.matmul(U_fw, expm(G_1))#non-unitary local gate in xy-model that causes eigenvalues to be complex. Contributes only for non-Ising couplings.
    U_bw = np.matmul(expm(G_1),U_bw)

    print('U_fw= ', U_fw)
    print('U_bw= ', U_bw)

    # G_eff is equivalent to generator for composed map (in principle obtainable through Baker-Campbell-Hausdorff)
    G_eff_fw = -1j * linalg.logm(U_fw)
    G_eff_bw = +1j * linalg.logm(U_bw)

    print('G_eff_fw = ')
    print(G_eff_fw)

    print('G_eff_bw = ')
    print(G_eff_bw)

    #add small random part to G_eff to lift degenaracies, such that numerical diagnoalization is more stable
    random_part = np.random.rand(2 * nsites, 2 * nsites) * 1e-10
    # symmetrize random part
    for i in range(2*nsites):
        for j in range(i, 2*nsites):
            random_part[i, j] = random_part[j, i]
    G_eff_fw += random_part
    G_eff_bw += random_part

    # compute eigensystem of G_eff. Set of eigenvectors "eigenvectors_G_eff_fw/bw" diagnonalizes G_eff_fw/bw
    eigenvalues_G_eff_fw = np.zeros(2 * nsites, dtype=np.complex_)
    eigenvalues_G_eff_bw = np.zeros(2 * nsites, dtype=np.complex_)
    eigenvectors_G_eff_fw = np.zeros((2 * nsites, 2 * nsites), dtype=np.complex_)
    eigenvectors_G_eff_bw = np.zeros((2 * nsites, 2 * nsites), dtype=np.complex_)

    if abs(Jy) < 1e-10 or abs(Jx) < 1e-10:
        eigenvalues_G_eff_fw, eigenvectors_G_eff_fw = linalg.eig(0.5 * (G_eff_fw + G_eff_fw.conj().T))#take superposition with hermitian conjugate to stabilize numerical diagonalization (works only in unitary case, e.g. Ising-type coupling)
        eigenvalues_G_eff_bw, eigenvectors_G_eff_bw = linalg.eig(0.5 * (G_eff_bw + G_eff_bw.conj().T))#take superposition with hermitian conjugate to stabilize numerical diagonalization (works only in unitary case, e.g. Ising-type coupling)

    else:
        eigenvalues_G_eff_fw, eigenvectors_G_eff_fw = linalg.eig(G_eff_fw)
        eigenvalues_G_eff_bw, eigenvectors_G_eff_bw = linalg.eig(G_eff_bw)

    #check if found eigenvectors indeed fulfill eigenvector equation (trivial check)
    eigenvector_check_fw = 0
    eigenvector_check_bw = 0
    for i in range(nsites):
        eigenvector_check_fw += linalg.norm(np.dot(G_eff_fw, eigenvectors_G_eff_fw[:, i]) - np.dot(eigenvalues_G_eff_fw[i], eigenvectors_G_eff_fw[:, i]))
        eigenvector_check_bw += linalg.norm(np.dot(G_eff_bw, eigenvectors_G_eff_bw[:, i]) - np.dot(eigenvalues_G_eff_bw[i], eigenvectors_G_eff_bw[:, i]))
    print 'eigenvector_check (f/b)', eigenvector_check_fw,'/',eigenvector_check_bw

    print 'forward eigenvalues'
    print eigenvalues_G_eff_fw

    print 'backward eigenvalues'
    print eigenvalues_G_eff_bw

    print 'forward eigenvectors'
    print eigenvectors_G_eff_fw

    print 'backward eigenvectors'
    print eigenvectors_G_eff_bw

    #sort eigenvectors such that first half are the ones with positive real part of eigenvalues and second half the corresponding negative ones
    argsort_fw = np.argsort(- np.real(eigenvalues_G_eff_fw))
    argsort_bw = np.argsort(- np.real(eigenvalues_G_eff_bw))
    M_fw = np.zeros((2 * nsites, 2 * nsites), dtype=np.complex_)
    M_bw = np.zeros((2 * nsites, 2 * nsites), dtype=np.complex_)
    for i in range(nsites):  # sort eigenvectors and eigenvalues such that the first half are the ones with positive real part, and the second half have negative real parts
        M_fw[:, i] = eigenvectors_G_eff_fw[:, argsort_fw[i]]
        M_fw[:, 2 * nsites - 1 - i] = eigenvectors_G_eff_fw[:, argsort_fw[i + nsites]]
        M_bw[:, i] = eigenvectors_G_eff_bw[:, argsort_bw[i]]
        M_bw[:, 2 * nsites - 1 - i] = eigenvectors_G_eff_bw[:, argsort_bw[i + nsites]]
    print 'M_forward'
    print(M_fw) #matrix that diagonalizes G_eff_fw/bw (equal on both branches)
    print 'M_backward'
    print(M_bw) #matrix that diagonalizes G_eff_fw/bw (equal on both branches)
    

    # diagonalize G_eff with eigenvectors to check:
    M_fw_inverse = linalg.inv(M_fw)
    M_bw_inverse = linalg.inv(M_bw)
    D_fw = np.dot(M_fw_inverse, G_eff_fw)
    D_fw = np.dot(D_fw, M_fw)# this is the diagonal matrix with eigenvalues of G_eff on the diagonal
    eigenvalues_G_eff_fw = D_fw.diagonal()# this makes sure that the order of the eigenvalues corresponds to the order of the eigenvectors in the matrix M
    print('D_fw= ')
    print(D_fw)

    D_bw = np.dot(M_bw_inverse, G_eff_bw)
    D_bw = np.dot(D_bw, M_bw)# this is the diagonal matrix with eigenvalues of G_eff on the diagonal
    eigenvalues_G_eff_bw = D_bw.diagonal()# this makes sure that the order of the eigenvalues corresponds to the order of the eigenvectors in the matrix M
    print('D_bw= ')
    print(D_bw)

    #check if diagonalization worked
    diag_check_fw = 0
    diag_check_bw = 0
    for i in range(0, 2 * nsites):
        for j in range(i + 1, 2 * nsites):
            diag_check_fw += abs(D_fw[i, j])
            diag_check_bw += abs(D_bw[i, j])
    print 'diag_checks (fw/bw)', diag_check_fw,'/', diag_check_bw 

    
    #eigenvalues_G_eff = ews_sorted
    print ('eigenvalues of G_eff_fw: ')
    print(eigenvalues_G_eff_fw)

    print ('eigenvalues of G_eff_bw: ')
    print(eigenvalues_G_eff_bw)

    print('Diagonalization of Generator completed..')
    return M_fw, M_fw_inverse, M_bw, M_bw_inverse,  eigenvalues_G_eff_fw, eigenvalues_G_eff_bw, eigenvalues_G_eff_fw.size / 2 
