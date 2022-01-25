import numpy as np
from scipy.sparse.dok import dok_matrix

def add_cmplx_random_antisym(matrix, magnitude):
    matrix = dok_matrix(matrix)
    print('add, dim', matrix.shape[0])
    if matrix.shape[0]!= matrix.shape[1]:
        print ('ERROR: Matrix is not a square matrix - no random part has been added.')

    else:    
        dim = matrix.shape[0] # matrix is square matrix with dimension dim
        random_part = np.random.rand(dim,dim) * magnitude 
        
        #antisymmetrize random part
        for i in range(dim):
            for j in range(i, dim):
                if i == j:
                    random_part[i, j] = 0   
                else:
                    random_part[j, i] = - random_part[i, j]
                    
        matrix += (random_part)
    
    return matrix
