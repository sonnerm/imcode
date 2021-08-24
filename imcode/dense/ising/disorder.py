import numpy as np
def popcount(i):
    return sum([int(k) for k in bin(i)[2:]])
def hr_operator(t,weights=None):
    if weights is None:
        weights=[1.0]+[0.0,0.0]*t
    weights=np.array(weights)
    ret=np.zeros(4**t,dtype=np.common_type(np.array(1.0),weights))
    evm=int("01"*t,2)
    odm=~evm
    for i in range(4**t):
        wi=popcount(i&evm)-popcount(i&odm)
        ret[i]=weights[2*abs(wi)+(-1 if wi>0 else 0)]
    return np.diag(ret)

def Jr_operator(t,weights=None):
    if weights is None:
        weights=[1.0]+[0.0,0.0]*t
    weights=np.array(weights)
    ret=np.zeros((4**t,4**t),dtype=np.common_type(np.array(1.0),weights))
    evm=int("01"*t,2)
    odm=~evm
    for i in range(4**t):
        wi=popcount(i&evm)-popcount(i&odm)
        for j in range(4**t):
            ret[j,j^i]=weights[2*abs(wi)+(-1 if wi>0 else 0)]
    return ret
