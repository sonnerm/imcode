import numpy as np
import scipy.linalg as la
def ising_H(J,g):
    if len(J)<len(g):
        J=np.array(list(J)+[0.0])
    J=np.array(J)
    g=np.array(g)
    diage=np.zeros(len(g)+len(J)-1,dtype=np.common_type(J,g))
    diage[::2]=g
    diage[1::2]=J[:-1]
    rete=0.5j*(np.diag(diage,1)-np.diag(diage,-1))
    if len(g)>1:
        rete[0,-1]+=-0.5j*J[-1]
        rete[-1,0]+=0.5j*J[-1]
    reto=0.5j*(np.diag(diage,1)-np.diag(diage,-1))
    if len(g)>1:
        reto[0,-1]+=0.5j*J[-1]
        reto[-1,0]+=-0.5j*J[-1]
    return (rete*4,reto*4)

def ising_F(J,g):
    if len(J)<len(g):
        J=list(J)+[0.0]
    J=np.array(J)
    g=np.array(g)
    diagg=np.zeros(len(g)+len(J)-1,dtype=np.common_type(J,g))
    diagg[::2]=g
    U1h=np.diag(diagg,1)-np.diag(diagg,-1)
    U1=la.expm(-U1h*2)
    diagJ=np.zeros(len(g)+len(J)-1,dtype=np.common_type(J,g))
    diagJ[1::2]=J[:-1]
    U2ho=np.diag(diagJ,1)-np.diag(diagJ,-1)
    U2ho[0,-1]-=J[-1]
    U2ho[-1,0]-=J[-1]
    U2he=np.diag(diagJ,1)-np.diag(diagJ,-1)
    U2he[0,-1]+=J[-1]
    U2he[-1,0]+=J[-1]
    U2o=la.expm(-U2ho*2)
    U2e=la.expm(-U2he*2)
    return (U2e@U1,U2o@U1)
def ising_T(J,g):
    pass
