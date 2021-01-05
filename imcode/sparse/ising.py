import numpy as np

def ising_diag(J,h):
    ret=np.zeros((2**len(hs)),dtype=complex)
    for i,h in enumerate(hs):
        cret=np.ones((2**i,2,2**(len(hs)-1-i)))/2
        cret[:,1,:]=-1/2
        ret+=np.ravel(cret*h)
    for i,J in enumerate(Js[:-1]):
        cret=np.ones((2**i,2,2,2**(len(Js)-2-i)))/4
        cret[:,1,0,:]=-1/4
        cret[:,0,1,:]=-1/4
        ret+=np.ravel(cret)*J
    cret=np.ones((2,2**(len(Js)-2),2))/4
    cret[1,:,0]=-1/4
    cret[0,:,1]=-1/4
    ret+=np.ravel(cret)*Js[-1]
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
