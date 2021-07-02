import numpy as np

def add_cmplx_random_antisym(matrix, magnitude):

    if len(matrix)!= len(matrix[0]):
        print ('ERROR: Matrix is not a square matrix - no random part has been added.')

    else:    
        dim = len(matrix) # matrix is square matrix with dimension dim
        random_part = np.random.rand(dim,dim) * magnitude * matrix[0,0]
        
        #antisymmetrize random part
        for i in range(dim):
            for j in range(i, dim):
                if i == j:
                    random_part[i, j] = 0   
                else:
                    random_part[j, i] = - random_part[i, j]
                    
        matrix += (random_part)
    
    return matrix