import numpy as np
def unitary_channel(F):
    L=int(np.log2(F.shape[0]))
    ret=np.kron(F,F.conj()).reshape((2,)*(4*L))
    tplist=sum(zip(range(L),range(L,2*L)),())+sum(zip(range(2*L,3*L),range(3*L,4*L)),())
    ret=ret.transpose(tplist)
    return ret.reshape((F.shape[0]**2,F.shape[1]**2))
