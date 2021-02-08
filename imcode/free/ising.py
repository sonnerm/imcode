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

# def etat(g):
#     return np.pi/4.0j+np.log(np.sin(g))/2+np.log(np.cos(g))/2
# def Jt(g):
#     return -np.pi/4-np.log(np.tan(g))*0.5j
# def gt(J):
#     return np.arctan(1.0j*np.exp(2j*J))
#
# def dualU(eta,J,g):
#     gn=gt(J)
#     Jn=Jt(g)
#     etan=eta+etat(g)-etat(gn)
#     return (etan,Jn,gn)

def ising_J(T,J):
    bd=np.array([[1,1.0j],[-1.0j,1]])/np.sqrt(2)
    fw=np.array([[np.exp(1.0*J),1.0j*np.exp(-1.0j*J)],[-1.0j*np.exp(-1.0j*J),np.exp(1.0*J)]])
    bw=np.array([[np.exp(-1.0*J),1.0j*np.exp(1.0j*J)],[-1.0j*np.exp(1.0j*J),np.exp(-1.0*J)]])
    ret=np.zeros((4*T,4*T),dtype=complex)
    ret[(0,1),:][:,(0,1)] = bd
    for i in range(1,T):
        ret[(2*i,2*i+1),:][:,(2*i,2*i+1)] = fw
    ret[(2*T,2*T+1),:][:,(2*T,2*T+1)] = bd
    for i in range(T+1,2*T):
        ret[(2*i,2*i+1),:][:,(2*i,2*i+1)] = bw
    return ret

def ising_W(T,g):
    fw=np.array([[np.cos(g),np.sin(g)],[-np.sin(g),np.cos(g)]])
    bw=np.array([[np.cos(g),-np.sin(g)],[np.sin(g),np.cos(g)]])
    ret=np.zeros((4*T,4*T))
    for i in range(T):
        ret[(2*i+1,2*i+2),:][:,(2*i,2*i+1)] = fw
    for i in range(T,2*T-1):
        ret[(2*i+1,2*i+2),:][:,(2*i,2*i+1)] = bw
    rete=np.copy(ret)
    rete[(0,-1),:][:,(0,-1)]=bw
    reto=ret
    reto[(0,-1),:][:,(0,-1)]=-bw
    return (rete,reto)

def ising_T(T,J,g):
    U1=ising_J(T,J)
    U2e,U2o=ising_W(T,g)
    return (U1@U2e,U1@U2o)
