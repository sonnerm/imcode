import numpy as np
from scipy import linalg
from scipy.sparse.linalg import eigsh
from add_cmplx_random_antisym import add_cmplx_random_antisym
np.set_printoptions(linewidth=np.nan, precision=2, suppress=True)

def find_index_dm(x,bar,N_l):
    return N_l * bar + x

def compute_Kernel_XX(beta, N_l):
    print('N_l',N_l)
#exponent of DM in OPERATOR form, i.e. e^{\xsi A \xsi} (no sign, no prefactor). Below, enter A
    A = np.zeros((2*N_l,2*N_l))

    measure = np.zeros((2*N_l,2*N_l))
    for i in range(N_l):
        measure[i+N_l, i] += 1
        measure[i, i+N_l] -= 1
    print(measure)

    beta2 = 2.
#thermal XX model:
    
    for i in range(N_l-1):
        
        A[i,i+1] += - beta#no factor 1/2 here since J_+ = 2 * J_x in XX model
        A[i+1,i] += - beta

        A[i + N_l,i + N_l +1] += beta
        A[i + N_l +1,i + N_l] += beta

       #XY (Jx neq Jy)
        #A[i+1,i+N_l] = - beta2#no factor 1/2 here since J_+ = 2 * J_x in XX model
        #A[i+N_l,i+1] = - beta2
        #A[i,i+N_l+1] =  beta2#no factor 1/2 here since J_+ = 2 * J_x in XX model
        #A[i+N_l+1,i] =  beta2
    
    
    #thermal product state e^{-\beta Z}
    #for i in range (N_l):
    #    A[i,i] = - beta
    #    A[i+N_l,i+N_l] = + beta

    
    #A = add_cmplx_random_antisym(A,1.e-8)
    #hermitian_matrix = hermitian_matrix.T @ hermitian_matrix.conj()
    #A += hermitian_matrix
    print('A')
    print(A)
    
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
    """
    print('M')
    print(M)
    print('daig')
    print(eigvals)
    print(M.T.conj() @ A @ M)
    """
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
                #dagger, no-dagger

                #sum21 += M[i, k].conj() * M[j, k] * np.exp(eigvals[k])/ Z_k
                #sum21 += M[i, k+N_l].conj() * M[j, k + N_l] * np.exp(eigvals[k + N_l]) / Z_k
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

            """
            corr_matrix[i,j] = sum11
            corr_matrix[i+N_l,j] = sum21
            corr_matrix[i,j+N_l] = sum12
            corr_matrix[i+N_l,j+N_l] = sum22
            """
    #corr_block[N_l:2*N_l,0:N_l] =  - corr_block[0:N_l,N_l:2*N_l].T

    """
    #check that correlations fulfill relations in correlaion matrix:
    check1 = corr_matrix[0:N_l,0:N_l] + corr_matrix[N_l:2*N_l,N_l:2*N_l].T - np.identity(N_l) 
    check2 = corr_matrix[0:N_l,N_l:2*N_l] - corr_matrix[N_l:2*N_l,0:N_l].T.conj()
    print('checks (must be zero)')
    print(check1)
    print(check2)
    print('corr block')
    print(corr_block)
    """

    """
    rand_magn = 0
    rand_A = np.random.rand(N_l,N_l) * rand_magn
    rand_B = add_cmplx_random_antisym(np.zeros((N_l,N_l)), rand_magn)
    rand_C = add_cmplx_random_antisym(np.zeros((N_l,N_l)), rand_magn)
    random_part = np.bmat([[rand_A,rand_B], [rand_C, -rand_A.T]])
    corr_block += random_part
    """

    B_inv = linalg.inv(corr_block) + measure
   
    #adjust signs
    for i in range(N_l):
        for j in range(N_l, 2*N_l):
            B_inv[i,j] = -B_inv[i,j]
            B_inv[j,i] = -B_inv[j,i]

    print('B_inv')
    return B_inv

