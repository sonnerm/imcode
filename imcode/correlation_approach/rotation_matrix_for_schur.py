import numpy as np
from scipy import linalg
def rotation_matrix_for_schur(B):#this funciton computes the orthogonal matrix that brings B into Schur form 

    dim_B = len(B)
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


    """
    #for general case:
    T, Z = linalg.schur(1j*B, output='complex')

    argsort2 = np.argsort(- np.sign(np.real(np.diag(T))) * np.abs(np.diag(T)))

    ves_sorted2 = np.zeros((dim_B, dim_B), dtype=np.complex_)
    ews_sorted2 = np.zeros(dim_B, dtype=np.complex_)

    for i in range(dim_B / 2):
        ves_sorted2[:, (2 * i) + 1] = (Z[:, argsort2[(dim_B) - 1 - i]])
        ves_sorted2[:, 2 * i] = (Z[:, argsort2[i]])
        ews_sorted2[(2 * i) + 1] = np.diag(T)[argsort2[(dim_B) - 1 - i]]
        ews_sorted2[2 * i] = np.diag(T)[argsort2[i]]

    for i in range(dim_B / 2):
        ves_sorted2[:, 2*i] *= np.exp(-1j * np.angle(ves_sorted2[0, 2*i]))
        ves_sorted2[:, 2*i + 1] *= np.exp(1j * np.angle(ves_sorted2[0, 2*i]/ves_sorted2[0, 2*i + 1]))
    ves_sorted2 *= np.exp(-1j * np.pi/4)

    B_schur = np.dot(ves_sorted2.T, B)
    B_schur = np.dot(G_schur, ves_sorted2)
    schur_check = 0

    for i in range(dim_B):
        for j in range(i, dim_B):
            schur_check += abs(B_schur[i, j])
    for i in range(dim_B / 2):
        schur_check -= ews_sorted[i]

    print('Check that R yields Schur form \n', schur_check, '\n B_schur\n')
    print(B_schur)
    """
    return R, eigenvalues_sorted