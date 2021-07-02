import numpy as np
from numpy.core.einsumfunc import einsum
from scipy import linalg
from scipy.sparse.linalg import eigsh
def rotation_matrix_for_schur(B):#this funciton computes the orthogonal matrix that brings B into Schur form 
    dim_B = len(B)
    """
    #Ising case:
    eigenvalues, eigenvectors = linalg.eig(np.dot(1j, B))

    argsort = np.argsort(- np.sign(np.real(eigenvalues)) * np.abs(eigenvalues))
    eigenvectors_sorted = np.zeros((dim_B,dim_B), dtype=np.complex_)
    eigenvalues_sorted = np.zeros(dim_B, dtype=np.complex_)

    for i in range(dim_B / 2):
        eigenvectors_sorted[:, i] = eigenvectors[:, argsort[i]]
        eigenvectors_sorted[:, dim_B - 1 - i] = eigenvectors[:, argsort[i + dim_B / 2]]
        eigenvalues_sorted[i] = eigenvalues[argsort[i]]
        eigenvalues_sorted[dim_B - 1 - i] = eigenvalues[argsort[i + dim_B / 2]]

    R = np.zeros((dim_B, dim_B))#this is the matrix that brings B into Schur form by R^T B R
    for i in range(0, dim_B / 2):
        R[:, 2 * i] = np.real(eigenvectors_sorted[:, i]) * 2**0.5#this work only in Ising case where B is real 
        R[:, 2 * i + 1] = np.imag(eigenvectors_sorted[:, i]) * 2**0.5#this work only in Ising case where B is real 

    schur = np.dot(R.T, B)
    schur = np.dot(schur, R)
    print 'Matrix B brought into Schur form:' ,schur


    #for general case:
    T, Z = linalg.schur(1j*B, output='complex')
    print np.diag(T)
    argsort2 = np.argsort(- np.sign(np.real(np.diag(T))) * np.abs(np.diag(T)))

    eigenvectors_sorted2 = np.zeros((dim_B, dim_B), dtype=np.complex_)
    eigenvalues_sorted2 = np.zeros(dim_B, dtype=np.complex_)
    
    for i in range(dim_B / 2):
        eigenvectors_sorted2[:, (2 * i) + 1] = (Z[:, argsort2[(dim_B) - 1 - i]])
        eigenvectors_sorted2[:, 2 * i] = (Z[:, argsort2[i]])
        eigenvalues_sorted2[(2 * i) + 1] = np.diag(T)[argsort2[(dim_B) - 1 - i]]
        eigenvalues_sorted2[2 * i] = np.diag(T)[argsort2[i]]
    
    for i in range(dim_B / 2):
        eigenvectors_sorted2[:, 2*i] *= np.exp(-1j * np.angle(eigenvectors_sorted2[0, 2*i]))
        eigenvectors_sorted2[:, 2*i + 1] *= np.exp(1j * np.angle(eigenvectors_sorted2[0, 2*i]/eigenvectors_sorted2[0, 2*i + 1]))
    eigenvectors_sorted2 *= np.exp(-1j * np.pi/4)
    
    B_schur = np.dot(eigenvectors_sorted2.T, B)
    B_schur = np.dot(B_schur, eigenvectors_sorted2)
    schur_check = 0

    for i in range(dim_B):
        for j in range(i, dim_B):
            schur_check += abs(B_schur[i, j])
    for i in range(dim_B / 2):
        schur_check -= eigenvalues_sorted2[i]
    print eigenvalues_sorted2
    print('Check that R yields Schur form \n', schur_check, '\n B_schur\n')
    print(B_schur)
    """

    hermitian_matrix = np.matmul(B.T, B.conj())

    random_part = np.random.rand(dim_B,dim_B) * 1e-10

    eigenvalues_hermitian_matrix, R = linalg.eigh(hermitian_matrix)#eigsh is not as reliable

    eigenvalues_B, eigenvectors_B = linalg.eig(1j*B)
    print(eigenvalues_B)
    B_schur = np.zeros((dim_B, dim_B), dtype=np.complex_)

    B_schur = einsum('ij,jk,kl->il',R.T.conj(),B,R.conj())

    print('Schur form of B \n')
    print(B_schur)
    B_check = np.zeros((dim_B, dim_B), dtype=np.complex_)
    B_schur_check = B_schur
    for i in range (dim_B):
        for j in range (dim_B):
            if abs(i-j) != 1 or i+j+1%4 == 0:
                B_schur_check[i,j] = 0

    print ('schur-test', linalg.norm(einsum('ij,jk,kl->il',R,B_schur_check,R.T) - B))
  
    return R, B_schur