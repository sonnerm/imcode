from add_cmplx_random_antisym import add_cmplx_random_antisym
from scipy import linalg
import numpy as np
def rotation_matrix_for_schur(B):#this funciton computes the orthogonal matrix that brings B into Schur form 
    dim_B = len(B)

    hermitian_matrix = add_cmplx_random_antisym(B,1.e-8)
    hermitian_matrix = B.T @ B.conj()
    eigenvalues_hermitian_matrix, R = linalg.eigh(hermitian_matrix)#compute rotation matrix as eigenvalues of hermitian_matrix, (eigsh is not as reliable)

    B_schur = R.T.conj() @ B @ R.conj()
    print('Schur form of B (complex eigenvalues):')
    print(B_schur)

    #check that B_schur is indeed in the desired Schur form
    B_schur_check = B_schur
    for i in range (dim_B):
        for j in range (dim_B):
            if abs(i-j) != 1 or i+j+1%4 == 0:
                B_schur_check[i,j] = 0
    print ('schur-test', linalg.norm(R @ B_schur_check @ R.T - B))

    eigenvalues_hermitian_matrix = np.zeros(int(dim_B/2), dtype=np.complex_)    
    for i in range(0,int(dim_B/2)):
        eigenvalues_hermitian_matrix[i] = B_schur[2 * i,2 * i + 1]#define eigenvalues from Schur form of matrix such that the order is in correspondence with the order of the eigenvectors in R.
    
    #this is the matrix that contains the phases, which can be absorbed in R, such that  R.T.conj() @ B @ R.conj() is real and all entries in the upper right triangular matrix are positive.
    D_phases = np.zeros((dim_B, dim_B), dtype=np.complex_)
    for i in range(int(dim_B/2)):
        D_phases[2 * i,2 * i] = np.exp(0.5j * np.angle(eigenvalues_hermitian_matrix[i]))
        D_phases[2 * i + 1,2 * i + 1] = np.exp(0.5j * np.angle(eigenvalues_hermitian_matrix[i]))

    R = R @ D_phases #R is generally complex
    print('rotation matrix R')
    print (R)
    B_schur = R.T.conj() @ B @ R.conj()
    print('Schur form of B (real eigenvalues):')
    print(B_schur)

    eigenvalues_real = np.zeros(int(dim_B/2))
    for i in range(0,int(dim_B/2)):
        eigenvalues_real[i] = B_schur[2 * i,2 * i + 1]#define eigenvalues from Schur form of matrix such that the order is in correspondence with the order of the eigenvectors in R.
    print ('eigenvalues (real): ',eigenvalues_real)


    return R,  eigenvalues_real #return rotation matrix R and schur form of B, as well as eigenvalues which are all real with the rotation matrix R