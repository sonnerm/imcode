import numpy as np
from scipy import linalg
from scipy.sparse.linalg import eigsh
np.set_printoptions(linewidth=np.nan, precision=2, suppress=True)

def find_index_dm(x,bar,N_l):
    return N_l * bar + x

def compute_Kernel(N_l):
#exponent of DM in OPERATOR form, i.e. e^{\xsi A \xsi} (no sign, no prefactor). Below, enter A
    A = np.zeros((2*N_l,2*N_l))

    measure = np.zeros((2*N_l,2*N_l))
    for i in range(N_l):
        measure[i+N_l, i] += 1
        measure[i, i+N_l] -= 1
    print(measure)

#thermal XX model:
    beta = 1.
    for i in range(N_l-1):
        A[i,i+1] = - beta#no factor 1/2 here since J_+ = 2 * J_x in XX model
        A[i + N_l,i + N_l +1] = beta
        A[i+1,i] = - beta
        A[i + N_l +1,i + N_l] = beta
    print(A)
    

    #find the matrix that diagonalizes A:
    eigvals, eigvecs= eigsh(A,len(A[0]))
    
    argsort = np.argsort(- np.real(eigvals))
    M = np.zeros((eigvecs.shape), dtype=np.complex_)

    for i in range(N_l):  # sort eigenvectors and eigenvalues such that the first half are the ones with positive real part, and the second half have negative real parts
        M[:, i] = eigvecs[:, argsort[i]]
        M[:, 2 * N_l - 1 - i] = eigvecs[:, argsort[i + N_l]]

    diag = M.T.conj() @ A @ M
    eigvals = diag.diagonal()
    print('daig')
    print(M.T.conj() @ A @ M)
    print(M)
    corr_block = np.zeros((2*N_l, 2*N_l),dtype=np.complex_)

    for i in range(N_l):
        for j in range(N_l):
            sum11 = 0
            sum12 = 0
            sum21 = 0
            sum22 = 0
            for k in range(N_l):
                Z_k = 2*np.cosh(eigvals[k])
                #dagger, no-dagger
                sum11 += M[i+N_l, k + N_l] * M[j, k] * np.exp(eigvals[k]) #/ Z_k
                sum11 += M[i+N_l, k] * M[j, k + N_l] * np.exp(eigvals[k + N_l])# / Z_k
                # no-dagger, dagger
                sum21 += M[i, k + N_l] * M[j+N_l, k] * np.exp(eigvals[k])#/ Z_k
                sum21 += M[i, k] * M[j+N_l, k + N_l] * np.exp(eigvals[k + N_l])#/ Z_k
                #nodagger, nodagger
                sum22 += M[i, k + N_l] * M[j, k] * np.exp(eigvals[k])#/ Z_k
                sum22 += M[i, k] * M[j, k + N_l] * np.exp(eigvals[k + N_l])#/ Z_k
                #dagger, dagger
                sum12 += M[i+N_l, k + N_l] * M[j+N_l, k] * np.exp(eigvals[k])#/ Z_k
                sum12 += M[i+N_l, k] * M[j+N_l, k + N_l] * np.exp(eigvals[k + N_l])#/ Z_k

            corr_block[j+ N_l,i] = -sum21
            corr_block[i+ N_l,j + N_l] = sum12
            corr_block[i,j+ N_l] = sum21
            corr_block[i,j] = sum22

    print(corr_block)
    B_inv = linalg.inv(-corr_block)
    print (B_inv)
    B_inv = B_inv - measure
    print (0.5 * B_inv)

    return B_inv

