import numpy as np

def ising_diag(J,h):
    L=len(h)
    ret=np.zeros((2**L),dtype=complex)
    for i,hv in enumerate(h):
        cret=np.ones((2**i,2,2**(L-1-i)))
        cret[:,1,:]=-1
        ret+=np.ravel(cret*hv)
    for i,Jv in enumerate(J[:L-1]):
        cret=np.ones((2**i,2,2,2**(L-2-i)))
        cret[:,1,0,:]=-1
        cret[:,0,1,:]=-1
        ret+=np.ravel(cret)*Jv
    if len(J)==L:
        cret=np.ones((2,2**(L-2),2))
        cret[1,:,0]=-1
        cret[0,:,1]=-1
        ret+=np.ravel(cret)*J[-1]
    return ret
def ising_H(J,g,h):
    '''
    '''
    pass
def ising_F(J,g,h):
    '''
    '''
def ising_T(T,J,g,h):
    pass

def ising_SFF(T,J,g,h):
    pass
