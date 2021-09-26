import numpy as np
from scipy import linalg
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
        A[i,i+1] = - beta/2.
        A[i + N_l,i + N_l +1] = beta/2.
        A[i+1,i] = - beta/2.
        A[i + N_l +1,i + N_l] = beta/2.
    #print(A)
    

    #find the matrix that diagonalizes A:
    eigvals, M= linalg.eig(A)

    #print(M.T.conj() @ A @ M)
    #print(M)
    corr_block = np.zeros((2*N_l, 2*N_l),dtype=np.complex_)

    for i in range(N_l):
        for j in range(N_l):
            sum11 = 0
            sum12 = 0
            for k in range(N_l):
                #dagger, no-dagger
                sum11 += M[i+N_l, k + N_l] * M[j, k] * np.exp(eigvals[k])
                sum11 += M[i+N_l, k] * M[j, k + N_l] * np.exp(eigvals[k + N_l])
                #dagger, dagger
                sum12 += M[i+N_l, k + N_l] * M[j+N_l, k] * np.exp(eigvals[k])
                sum12 += M[i+N_l, k] * M[j+N_l, k + N_l] * np.exp(eigvals[k + N_l])
            corr_block[i+ N_l,j] = sum11
            corr_block[i+ N_l,j + N_l] = sum12
    corr_block[0:N_l, N_l:2*N_l] = corr_block[ N_l:2*N_l,0:N_l].T.conj()
    corr_block[0:N_l, 0:N_l] = np.identity(N_l) - corr_block[N_l:2*N_l, N_l:2*N_l].T

    B_inv = linalg.inv(-corr_block)
    B_inv = B_inv - measure

    return B_inv

