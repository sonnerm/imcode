import numpy as np
from numpy.core.numeric import identity
from scipy import linalg
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg.matfuncs import expm
from add_cmplx_random_antisym import add_cmplx_random_antisym
np.set_printoptions(linewidth=np.nan, precision=2, suppress=True)

def find_index_dm(x,bar,N_l):
    return N_l * bar + x

def compute_Kernel_XX(beta, N_l):

#exponent of DM in OPERATOR form, i.e. e^{\xsi A \xsi} (no sign, no prefactor). Below, enter A
    A = np.zeros((2*N_l,2*N_l))

    measure = np.zeros((2*N_l,2*N_l))
    for i in range(N_l):
        measure[i+N_l, i] += 1
        measure[i, i+N_l] -= 1

#thermal XX model:
    
    for i in range(N_l-1):
        
        A[i,i+1] += - beta#no factor 1/2 here since J_+ = 2 * J_x in XX model
        A[i+1,i] += - beta

        A[i + N_l,i + N_l +1] += beta
        A[i + N_l +1,i + N_l] += beta

    #thermal product state e^{-\beta Z}
    #for i in range (N_l):
    #    A[i,i] = - beta
    #    A[i+N_l,i+N_l] = + beta
    
    #find the matrix that diagonalizes A:
    eigvals, eigvecs= eigsh(A,len(A[0]))
    
    argsort = np.argsort(- np.real(eigvals))
    M = np.zeros((eigvecs.shape), dtype=np.complex_)

    if np.abs(beta - 0.0) <1e-8:
        M = np.identity(2*N_l)
    else:
        for i in range(N_l):  # sort eigenvectors and eigenvalues such that the first half are the ones with positive real part, and the second half have negative real parts
            M[:, i] = eigvecs[:, argsort[i]]
            M[:, 2 * N_l - 1 - i] = eigvecs[:, argsort[i + N_l]]
    
    diag = M.T.conj() @ A @ M
    eigvals = diag.diagonal()

    corr_block = np.zeros((2*N_l, 2*N_l),dtype=np.complex_)
    #corr_matrix = np.zeros((2*N_l, 2*N_l),dtype=np.complex_)

    for i in range(N_l):
        for j in range(N_l):
            sum11 = 0
            sum12 = 0
            sum21 = 0
            sum22 = 0
            for k in range(N_l):
                Z_k = 2*np.cosh(eigvals[k])

                # no-dagger, dagger
                sum12 += M[i+ N_l, k].conj() * M[j+N_l, k] * np.exp(eigvals[k])/ Z_k
                sum12 += M[i+N_l, k+ N_l].conj() * M[j+ N_l, k+ N_l] * np.exp(eigvals[k + N_l])/ Z_k
                #nodagger, nodagger
                sum11 += M[i+ N_l, k ].conj() * M[j, k] * np.exp(eigvals[k])/ Z_k
                sum11 += M[i, k] * M[j+ N_l, k ].conj() * np.exp(eigvals[k + N_l])/ Z_k
                #dagger, dagger
                sum22 += M[i, k].conj() * M[j+N_l, k] * np.exp(eigvals[k])/ Z_k
                sum22 += M[i+N_l, k] * M[j, k].conj() * np.exp(eigvals[k + N_l])/ Z_k

            corr_block[j+ N_l,i] = sum12 #correct
            corr_block[i+ N_l,j + N_l] = - sum22
            corr_block[i,j+ N_l] = - sum12 #correct
            corr_block[i,j] = - sum11

    print('correlations_block')
    print(corr_block)
    B_inv = linalg.inv(corr_block) + measure
    print('B_inv')
    print((B_inv-measure)@ corr_block)
    #adjust signs
    for i in range(N_l):
        for j in range(N_l, 2*N_l):
            B_inv[i,j] = -B_inv[i,j]
            B_inv[j,i] = -B_inv[j,i]

 
    return B_inv


def compute_gate_Kernel(Jx,Jy,g):
    Jp = (Jx+Jy)/2
    Jm = (Jy-Jx)/2

    if abs(abs(Jp) - abs(Jm))< 1e-5 and abs(g)<1e-5:
        g = 1.e-5
   
    
#exponent of DM in OPERATOR form, i.e. e^{\xsi A \xsi} (no sign, no prefactor). Below, enter A
    A = np.zeros((4,4),dtype=np.complex_)
    K = np.zeros((4,4),dtype=np.complex_)

    measure = np.zeros((4,4))
    for i in range(2):
        measure[i+2, i] += 1
        measure[i, i+2] -= 1

    stabilizer = np.zeros((4,4),dtype=np.complex_)
    tiny = 1.e-6 * 1j
    stabilizer[0,0] = 1.*tiny
    stabilizer[1,1] = 1.*tiny
    stabilizer[2,2] = -1.*tiny
    stabilizer[3,3] = -1.*tiny

    A[0,1] += 1.j * Jp#no factor 1/2 here since J_+ = 2 * J_x in XX model
    A[1,0] += 1.j * Jp
    A[2,3] -= 1.j * Jp
    A[3,2] -= 1.j * Jp
    
    A[0,3] -= 1.j * Jm#no factor 1/2 here since J_+ = 2 * J_x in XX model
    A[1,2] += 1.j * Jm
    A[2,1] += 1.j * Jm
    A[3,0] -= 1.j * Jm
    
    #Kick
    for i in range (2):
        K[i,i] = -1.j*g
        K[i+2,i+2] = 1.j*g
 
   
    A_eff = K + stabilizer
    #find the matrix that diagonalizes A_eff:
    eigvals, eigvecs= linalg.eigh(-1.j*A_eff)
    
    argsort = np.argsort(- np.real(eigvals))
    M = np.zeros((eigvecs.shape), dtype=np.complex_)

    
    for i in range(2):  # sort eigenvectors and eigenvalues such that the first half are the ones with positive real part, and the second half have negative real parts
        M[:, i] = eigvecs[:, argsort[i]]
        M[:, 3 - i] = eigvecs[:, argsort[i +2]]
 
    diag = M.T.conj() @ A_eff @ M
    eigvals = diag.diagonal()

    corr_block = np.zeros((4,4),dtype=np.complex_)

    for i in range(2):
        for j in range(2):
            sum11 = 0
            sum12 = 0
            sum21 = 0
            sum22 = 0
            for k in range(2):
                Z_k = 2*np.cosh(eigvals[k])
    
                # no-dagger, dagger
                sum12 += M[i+ 2, k].conj() * M[j+2, k] * np.exp(eigvals[k])/ Z_k
                sum12 += M[i+2, k+ 2].conj() * M[j+ 2, k+ 2] * np.exp(eigvals[k + 2])/ Z_k
                #nodagger, nodagger
                sum11 += M[i+ 2, k ].conj() * M[j, k] * np.exp(eigvals[k])/ Z_k
                sum11 += M[i, k] * M[j+ 2, k ].conj() * np.exp(eigvals[k + 2])/ Z_k
                #dagger, dagger
                sum22 += M[i, k].conj() * M[j+2, k] * np.exp(eigvals[k])/ Z_k
                sum22 += M[i+2, k] * M[j, k].conj() * np.exp(eigvals[k + 2])/ Z_k

            corr_block[j+ 2,i] = sum12 #correct
            corr_block[i+ 2,j + 2] = - sum22
            corr_block[i,j+ 2] = - sum12 #correct
            corr_block[i,j] = - sum11

    B_inv = linalg.inv(corr_block) + measure
   
    #adjust signs
    for i in range(2):
        for j in range(2, 4):
            B_inv[i,j] = -B_inv[i,j]
            B_inv[j,i] = -B_inv[j,i]

    return B_inv
