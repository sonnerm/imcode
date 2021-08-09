import numpy as np
def hr_operator(t,weights=None):
    if weights is None:
        weights=[1.0]+[0.0,0.0]*t
    ret=np.zeros(4**T)
    evm=int("01"*T,2)
    odm=~evm
    for i in range(4**(4*T)):
        wi=popcount(i&evm)-popcount(i&odm)
        ret[i]=weights[2*abs(wi)+(-1 if wi>0 else 0)]
    return np.diag(ret)

def Jr_operator(t,weights=None):
    if weights is None:
        weights=[1.0]+[0.0,0.0]*t
    ret=np.zeros((2**(2*T),2**(2*T)))
    for i in range(2**(2*T)):
        wi=popcount((i>>T)&(~(1<<(T-1))))-popcount((i^((i>>T)<<T))&(~(1<<(T-1))))
        for j in range(2**(2*T)):
            ret[j,j^i]=weights[2*abs(wi)+(-1 if wi>0 else 0)]
    return ret
