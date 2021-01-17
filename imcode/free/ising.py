import numpy as np
import numpy.linalg as la
def ising_H(J,g):
    if len(J)<len(g):
        J=list(J)+[0.0]
    rete=0.5j*(np.diag(np.ravel(np.zip(g,J[:-1])),1)+np.diag(np.ravel(np.zip(g,J[:-1])),-1))
    rete[0,-1]=0.5j*J[-1]
    rete[-1,0]=0.5j*J[-1]
    reto=0.5j*(np.diag(np.ravel(np.zip(g,J[:-1])),1)+np.diag(np.ravel(np.zip(g,J[:-1])),-1))
    reto[0,-1]=-0.5j*J[-1]
    reto[-1,0]=-0.5j*J[-1]
    return (rete,reto)

def ising_F(J,g):
    if len(J)<len(g):
        J=list(J)+[0.0]
    U1h=np.diag(np.ravel(np.zip(g,np.zeros(len(g)-1))),1)+np.diag(np.ravel(np.zip(g,np.zeros(len(g)-1))),-1)
    U1=la.expm(-0.5*U1h)
    U2ho=np.diag(np.ravel(np.zip(np.zeros(len(g)-1),J)),1)+np.diag(np.ravel(np.zip(g,np.zeros(len(g)-1))),-1)
    U2ho[0,-1]=-J[-1]
    U2ho[-1,0]=-J[-1]
    U2he=np.diag(np.ravel(np.zip(np.zeros(len(g)-1),J)),1)+np.diag(np.ravel(np.zip(g,np.zeros(len(g)-1))),-1)
    U2he[0,-1]=J[-1]
    U2he[-1,0]=J[-1]
    U2o=la.expm(-0.5*U2ho)
    U2e=la.expm(-0.5*U2he)
    return (U2e@U1,U2o@U1)
