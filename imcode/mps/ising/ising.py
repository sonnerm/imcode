from .. import MPO
import numpy as np
def ising_W(t,g,init=(0.5,0.5)):
    s2=np.abs(np.sin(g))**2
    c2=np.abs(np.cos(g))**2
    mx=-1.0j*np.conj(np.sin(g))*np.cos(g)
    px=1.0j*np.sin(g)*np.conj(np.cos(g))
    Wprim=np.array([[c2,px,mx,s2],
                    [px,c2,s2,mx],
                    [mx,s2,c2,px],
                    [s2,mx,px,c2]
    ])
    init=np.diag(list(init)+[0,0])
    W_0a=np.einsum("cd,cb,ac->abcd",init,np.eye(4),np.eye(1))
    W_ia=np.einsum("cd,cb,ac->abcd",np.eye(4),np.eye(4),Wprim)
    W_Ta=np.einsum("cd,cb,ac->abcd",np.eye(4),np.eye(1),Wprim)
    return MPO.from_matrices([W_0a]+[W_ia]*(t-1)+[W_Ta])
#TODO refactor

def ising_h(t,h):
    Ida=np.einsum("ab,cd->abcd",np.eye(1),np.eye(4))
    Ha=np.einsum("ab,cd->abcd",np.eye(1),np.diag([np.exp(1.0j*(h-np.conj(h))),np.exp(-1.0j*(h-np.conj(h))),np.exp(1.0j*(h+np.conj(h))),np.exp(-1.0j*(h+np.conj(h)))]))
    return MPO.from_matrices([Ida]+[Ha]*(t-1)+[Ida])

def ising_J(t,J):
    Iprim=np.array([[1.0,1.0,0.0,0.0],[1.0,1.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]])#/np.sqrt(2)
    Ida=np.einsum("ab,cd->abcd",np.eye(1),Iprim)
    pj=np.exp(2.0j*J)
    mj=np.exp(-2.0j*np.conj(J))
    ip=np.exp(1.0j*(J-np.conj(J)))
    Jprim=np.array([[ip,mj,pj,ip],
                    [mj,ip,ip,pj],
                    [pj,ip,ip,mj],
                    [ip,pj,mj,ip]
    ])
    Ja=np.einsum("ab,cd->abcd",np.eye(1),Jprim)
    return MPO.from_matrices([Ida]+[Ja]*(t-1)+[Ida])
def ising_T(t,J,g,h,init=np.eye(2)/2):
    return (ising_J(t,J)@ising_W(t,g,init)@ising_h(t,h)).contract()
