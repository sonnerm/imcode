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

    eps = 1e-8 # lift degeneracy
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
    U_E = np.zeros((2, 2 * nsites, 2 * nsites), dtype=np.complex_)#gate that describes evolution non disconnected environment. first dimension: 0 = forward branch, 1 = backward branch 
    U_eff = np.zeros((2, 2 * nsites, 2 * nsites), dtype=np.complex_)#gate that governs time evolution on both branches. first dimension: 0 = forward branch, 1 = backward branch 
    
    if abs(Jy) < 1e-10 or abs(Jx) < 1e-10:#for Ising-type couplings, the even and odd gates commute and can be added in the exponent (trivial Backer-Campbell Haussdorff)
        G_XY = G_XY_even + G_XY_odd
        U_E[0] = expm(1j * G_XY)
        U_E[1] = expm(-1j * G_XY)
    else:#for xy model, even and odd gates do not commute 
        U_E[0] = np.matmul(expm(1.j * G_XY_even), expm(1.j * G_XY_odd))
        U_E[1] = np.matmul(expm(-1.j * G_XY_odd), expm(-1.j * G_XY_even))


    #onsite kicks (used in KIC):
    U_E[0] = np.matmul(expm(1j*G_g), U_E[0])#this has an effect only when local onsite kicks (as in KIC) are nonzero
    U_E[1] = np.matmul(U_E[1], expm(-1j*G_g))


    #generator of environment (always unitary)
    G_eff_E = -1j * linalg.logm(U_E[0])

    #non-unitary gate stemming from vacuum projections (note that there is no imaginary j in from of G_1)
    U_eff[0] = np.matmul(U_E[0], expm(G_1))#non-unitary local gate in xy-model that causes eigenvalues to be complex. Contributes only for non-Ising couplings.
    U_eff[1] = np.matmul(expm(G_1),U_E[1])

    print('U_fw= ', U_eff[0])
    print('U_bw= ', U_eff[1])

    # G_eff is equivalent to generator for composed map (in principle obtainable through Baker-Campbell-Hausdorff)
    G_eff = np.zeros((2, 2 * nsites, 2 * nsites), dtype=np.complex_)
    G_eff[0] = -1j * linalg.logm(U_eff[0])
    G_eff[1] = +1j * linalg.logm(U_eff[1])

    print('G_eff_fw = ')
    print(G_eff[0])

    print('G_eff_bw = ')
    print(G_eff[1])

    #add small random part to G_eff to lift degenaracies, such that numerical diagnoalization is more stable
    random_part = np.random.rand(2 * nsites, 2 * nsites) * 1e-10
    # symmetrize random part
    for i in range(2*nsites):
        for j in range(i, 2*nsites):
            random_part[i, j] = random_part[j, i]
    G_eff[0] += random_part
    G_eff[1] += random_part
    G_eff_E += random_part

    # compute eigensystem of G_eff. Set of eigenvectors "eigenvectors_G_eff_fw/bw" diagnonalizes G_eff_fw/bw
    eigenvalues_G_eff = np.zeros((2, 2 * nsites), dtype=np.complex_)#first dimension: foward branch (index 0) and backward branch (index 1)
    #eigenvalues_G_eff_bw = np.zeros(2 * nsites, dtype=np.complex_)
    eigenvalues_G_eff_E = np.zeros(2 * nsites, dtype=np.complex_)#no need for seperate forward and backward branch since F_E is always unitary, time evolution is equivalent on both branches
    eigenvectors_G_eff = np.zeros((2, 2 * nsites, 2 * nsites), dtype=np.complex_)#first dimension: foward branch (index 0) and backward branch (index 1)
    #eigenvectors_G_eff_bw = np.zeros((2 * nsites, 2 * nsites), dtype=np.complex_)
    eigenvectors_G_eff_E = np.zeros((2 * nsites, 2 * nsites), dtype=np.complex_)

    if abs(Jy) < 1e-10 or abs(Jx) < 1e-10:
        eigenvalues_G_eff[0], eigenvectors_G_eff[0] = linalg.eig(0.5 * (G_eff[0] + G_eff[0].conj().T))#take superposition with hermitian conjugate to stabilize numerical diagonalization (works only in unitary case, e.g. Ising-type coupling)
        eigenvalues_G_eff[1], eigenvectors_G_eff[1] = linalg.eig(0.5 * (G_eff[1] + G_eff[1].conj().T))#take superposition with hermitian conjugate to stabilize numerical diagonalization (works only in unitary case, e.g. Ising-type coupling)
        eigenvalues_G_eff_E, eigenvectors_G_eff_E = linalg.eig(0.5 * (G_eff_E + G_eff_E.conj().T))#take superposition with hermitian conjugate to stabilize numerical diagonalization (works only in unitary case, e.g. Ising-type coupling)


    else:
        eigenvalues_G_eff[0], eigenvectors_G_eff[0] = linalg.eig(G_eff[0])
        eigenvalues_G_eff[1], eigenvectors_G_eff[1] = linalg.eig(G_eff[1])
        #eigenvalues_G_eff_bw, eigenvectors_G_eff_bw = linalg.eig(G_eff_bw)
        eigenvalues_G_eff_E, eigenvectors_G_eff_E = linalg.eig(G_eff_E)

    #check if found eigenvectors indeed fulfill eigenvector equation (trivial check)
    eigenvector_check = 0
    #eigenvector_check_bw = 0
    eigenvector_check_E = 0
    for branch in range (0,2):
        for i in range(nsites):
            eigenvector_check += linalg.norm(np.dot(G_eff[branch], eigenvectors_G_eff[branch,:, i]) - np.dot(eigenvalues_G_eff[branch,i], eigenvectors_G_eff[branch,:, i]))
        #eigenvector_check_bw += linalg.norm(np.dot(G_eff_bw, eigenvectors_G_eff_bw[:, i]) - np.dot(eigenvalues_G_eff_bw[i], eigenvectors_G_eff_bw[:, i]))
    for i in range(nsites):
        eigenvector_check_E += linalg.norm(np.dot(G_eff_E, eigenvectors_G_eff_E[:, i]) - np.dot(eigenvalues_G_eff_E[i], eigenvectors_G_eff_E[:, i]))
    print 'eigenvector_check (f/b/E)', eigenvector_check,'/',eigenvector_check_E

    print 'forward eigenvalues'
    print eigenvalues_G_eff[0]

    print 'backward eigenvalues'
    print eigenvalues_G_eff[1]

    print 'environment eigenvalues'
    print eigenvalues_G_eff_E

    #sort eigenvectors such that first half are the ones with positive real part of eigenvalues and second half the corresponding negative ones
    argsort_fw = np.argsort(- np.real(eigenvalues_G_eff[0]))
    argsort_bw = np.argsort(- np.real(eigenvalues_G_eff[1]))
    #argsort_bw = np.argsort(- np.real(eigenvalues_G_eff_bw))
    argsort_E = np.argsort(- np.real(eigenvalues_G_eff_E))
    M = np.zeros((2,2 * nsites, 2 * nsites), dtype=np.complex_)
    #M_bw = np.zeros((2 * nsites, 2 * nsites), dtype=np.complex_)
    M_E = np.zeros((2 * nsites, 2 * nsites), dtype=np.complex_)
    for i in range(nsites):  # sort eigenvectors and eigenvalues such that the first half are the ones with positive real part, and the second half have negative real parts
        M[0,:, i] = eigenvectors_G_eff[0,:, argsort_fw[i]]
        M[0,:, 2 * nsites - 1 - i] = eigenvectors_G_eff[0,:, argsort_fw[i + nsites]]
        M[1,:, i] = eigenvectors_G_eff[1,:, argsort_bw[i]]
        M[1,:, 2 * nsites - 1 - i] = eigenvectors_G_eff[1,:, argsort_bw[i + nsites]]
        M_E[:, i] = eigenvectors_G_eff_E[:, argsort_E[i]]
        M_E[:, 2 * nsites - 1 - i] = eigenvectors_G_eff_E[:, argsort_E[i + nsites]]
    print 'M_forward'
    print(M[0]) #matrix that diagonalizes G_eff_fw
    print 'M_backward'
    print(M[1]) #matrix that diagonalizes G_eff_bw 
    print 'M_environment'
    print(M_E) #matrix that diagonalizes G_eff_E
    
    M_inverse = np.zeros((2,2 * nsites, 2 * nsites), dtype=np.complex_)
    # diagonalize G_eff with eigenvectors to check:
    M_inverse[0] = linalg.inv(M[0])
    M_inverse[1] = linalg.inv(M[1])
    M_E_inverse = linalg.inv(M_E)

    D = np.zeros((2,2 * nsites, 2 * nsites), dtype=np.complex_)
    for branch in range (2):
        D[branch] = np.dot(M_inverse[branch], G_eff[branch])
        D[branch] = np.dot(D[branch], M[branch])# this is the diagonal matrix with eigenvalues of G_eff on the diagonal
        eigenvalues_G_eff[branch] = D[branch].diagonal()# this makes sure that the order of the eigenvalues corresponds to the order of the eigenvectors in the matrix M

    D_E = np.dot(M_E_inverse, G_eff_E)
    D_E = np.dot(D_E, M_E)# this is the diagonal matrix with eigenvalues of G_eff on the diagonal
    eigenvalues_G_eff_E = D_E.diagonal()# this makes sure that the order of the eigenvalues corresponds to the order of the eigenvectors in the matrix M
    
    print('D_fw= ')
    print(D[0])
    print('D_bw= ')
    print(D[1])
    print('D_E= ')
    print(D_E)

    #check if diagonalization worked
    diag_check = 0
    #diag_check_bw = 0
    diag_check_E = 0
    for branch in range (2):
        for i in range(0, 2 * nsites):
            for j in range(i + 1, 2 * nsites):
                diag_check += abs(D[branch,i, j])
    for i in range(0, 2 * nsites):
        for j in range(i + 1, 2 * nsites):
            diag_check_E += abs(D_E[i, j])
    print 'diag_checks (fw+bw/E)', diag_check,'/', diag_check_E


    f = 0
    for k in range(nsites):
        f += abs(M_E[0, k])**2 - abs(M_E[nsites, k])**2 + 2j * imag(M_E[0, k]*M_E[nsites, k].conj())
    
    #eigenvalues_G_eff = ews_sorted
    print ('eigenvalues of G_eff_fw: ')
    print(eigenvalues_G_eff[0])

    print ('eigenvalues of G_eff_bw: ')
    print(eigenvalues_G_eff[1])

    #print ('eigenvalues of G_eff_bw: ')
    #print(eigenvalues_G_eff_bw)

    print ('eigenvalues of G_eff_E: ')
    print(eigenvalues_G_eff_E)

    print('Diagonalization of generators completed..')
    #return M_fw, M_fw_inverse, M_bw, M_bw_inverse,  eigenvalues_G_eff_fw, eigenvalues_G_eff_bw, f
    return M, M_inverse,eigenvalues_G_eff,  f
