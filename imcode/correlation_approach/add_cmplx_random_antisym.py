import numpy as np

def add_cmplx_random_antisym(matrix, magnitude):

    if len(matrix)!= len(matrix[0]):
        print 'ERROR: Matrix is not a square matrix - no random part has been added.'

    else:    
        dim = len(matrix) # matrix is square matrix with dimension dim
        random_part_real = np.random.rand(dim,dim) * magnitude
        random_part_imag = np.random.rand(dim,dim) * magnitude * 1j

        #antisymmetrize random part
        for i in range(dim):
            for j in range(i, dim):
                if i == j:
                    random_part_real[i, j] = 0
                    random_part_imag[i, j] = 0
                else:
                    random_part_real[j, i] = - random_part_real[i, j]
                    random_part_imag[j, i] = - random_part_imag[i, j]

        matrix += (random_part_real)

    return matrix