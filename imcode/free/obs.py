import numpy.linalg as la
def direct_czz(F,t,i,j):
    if i==0 and j==0:
        return la.matrix_power(F[0],t)[0,0]+la.matrix_power(F[1],t)[0,0]

def direct_cxx(F,t,i,j):
    la.matrix_power(F[0],t)+la.matrix_power(F[1],t)

def embedded_czz(left_im,lop,right_im):
    pass

def boundary_czz(im,lop):
    pass

def fold_entropy(im):
    pass

def flat_entropy(im):
    pass
