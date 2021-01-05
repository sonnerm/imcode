import transfer_dual_keldysh as tk
import numpy as np

def get_ladderF_dense(T,Jl,J,g,h):
    sec=tk.trivial_sector(4*T)
    Jt,_,eta1,_=tk.dualU(J,g)
    # h=np.array([0]+[2*h]*(T-1)+[0]+[-2*np.array(h).conj()]*(T-1))
    # Jl=np.array([4*Jl,0]*(T)+[-4*Jl.conj(),0]*T)
    # U1=np.exp(-1.0j*np.array(tk.get_imbrie_p(h,np.zeros_like(h),Jl).diagonal()))
    id=np.eye(2)
    hm=np.array([[h,0],[0,-h]])
    jlm=np.array([[Jl,0],[0,-Jl]])
    sz=np.array([[1,0],[0,-1]])
    jtm=np.array([[Jt,0],[0,-Jt]])
    U1=np.zeros((2**(4*T),2**(4*T)),dtype=complex)
    for i in range(1,T):
        U1+=tk.dense_kron([id,id]*i+[id,hm]+[id,id]*(2*T-1-i))
        U1+=tk.dense_kron([id,id]*i+[hm,id]+[id,id]*(2*T-1-i))
        U1+=tk.dense_kron([id,id]*i+[jlm,sz]+[id,id]*(2*T-1-i))

    for i in range(1,T):
        U1-=tk.dense_kron([id,id]*(i+T)+[id,hm]+[id,id]*(T-1-i))
        U1-=tk.dense_kron([id,id]*(i+T)+[hm,id]+[id,id]*(T-1-i))
        U1-=tk.dense_kron([id,id]*(i+T)+[jlm,sz]+[id,id]*(T-1-i))
    for i in range(T):
        U1+=tk.dense_kron([id,id]*i+[jtm,id]+[sz,id]+[id,id]*(2*T-2-i))
        U1+=tk.dense_kron([id,id]*i+[id,jtm]+[id,sz]+[id,id]*(2*T-2-i))

    for i in range(T-1):
        U1-=tk.dense_kron([id,id]*(T+i)+[jtm.conj(),id]+[sz,id]+[id,id]*(T-2-i))
        U1-=tk.dense_kron([id,id]*(T+i)+[id,jtm.conj()]+[id,sz]+[id,id]*(T-2-i))
    U1-=tk.dense_kron([sz,id]+[id,id]*(2*T-2)+[jtm.conj(),id])
    U1-=tk.dense_kron([id,sz]+[id,id]*(2*T-2)+[id,jtm.conj()])
    U1=np.diag(np.exp(-1.0j*np.diag(U1)))
    # # print(np.diag(U1))
    U1*=np.exp(eta1.real*(4*T))
    #
    Pm=np.array([[1,1],[1,1]])
    Tm1=np.array([[np.exp(1.0j*J),np.exp(-1.0j*J)],[np.exp(-1.0j*J),np.exp(1.0j*J)]])
    Tm2=Tm1.conj()
    U2=tk.dense_kron([Pm,Pm]+[Tm1,Tm1]*(T-1)+[Pm,Pm]+[Tm2,Tm2]*(T-1))/4
    ret=tk.project(U1@U2,sec)
    print(np.trace(ret))
    return ret
T=4;Jl=0.2;J=0.3;h=0.25;g=0.1

def get_ladder_boundary(T):
    ret=np.zeros((2**(4*T)))
    ret=ret.reshape((4,(2**(2*T-2)),4,(2**(2*T-2))))
    ret[0,:,0,:]=np.eye(2**(2*T-2))
    ret[0,:,1,:]=np.eye(2**(2*T-2))
    ret[0,:,2,:]=np.eye(2**(2*T-2))
    ret[0,:,3,:]=np.eye(2**(2*T-2))
    ret[1,:,0,:]=np.eye(2**(2*T-2))
    ret[1,:,1,:]=np.eye(2**(2*T-2))
    ret[1,:,2,:]=np.eye(2**(2*T-2))
    ret[1,:,3,:]=np.eye(2**(2*T-2))
    ret[2,:,0,:]=np.eye(2**(2*T-2))
    ret[2,:,1,:]=np.eye(2**(2*T-2))
    ret[2,:,2,:]=np.eye(2**(2*T-2))
    ret[2,:,3,:]=np.eye(2**(2*T-2))
    ret[3,:,0,:]=np.eye(2**(2*T-2))
    ret[3,:,1,:]=np.eye(2**(2*T-2))
    ret[3,:,2,:]=np.eye(2**(2*T-2))
    ret[3,:,3,:]=np.eye(2**(2*T-2))
    return np.ravel(ret)
def get_ladderF_op(T,Jl,J,g,h):
    Jt,gt,eta1,eta2=tk.dualU(J,g)
    gt=np.array([0.0,0.0]+[2*gt.conj()]*(2*T-2)+[0.0,0.0]+[-2*gt]*(2*T-2))
    D2=np.exp(1.0j*tk.get_imbrie_diag(gt,np.zeros_like(gt)))
    D2*=np.exp(-(4*T-4)*eta2.real)
    D2/=4
    D2*=get_ladder_boundary(T)
    # print(gt,eta2,D2)
    # D2=np.ones_like(D2)/2

    h=np.array([0.0,0.0]+[2*h]*(2*T-2)+[0.0,0.0]+[-2*np.array(h).conj()]*(2*T-2))
    Jt=np.array([0.0,0.0]+[4*Jl,0.0]*(2*T-2)+[0.0,0.0]+[-4*Jl.conj(),0.0]*(2*T-2))
    D1=np.exp(-1.0j*get_imbrie_diag(h,Jt))
    for i in range(2*T):
        D1=D1.reshape((2**i,4,2,2,2,2,4**(4*T-i)))
        D1[:,1,:,1,:,:]+=J
        D1[:,0,:,0,:,:]+=J
        D1[:,1,:,0,:,:]-=J
        D1[:,0,:,1,:,:]-=J

    for i in range(2*T):
        D1=D1.reshape((2**i,4,2,2,2,2,4**(4*T-i)))
        D1[:,1,:,1,:,:]+=J
        D1[:,0,:,0,:,:]+=J
        D1[:,1,:,0,:,:]-=J
        D1[:,0,:,1,:,:]-=J


    D1*=np.exp(2*T*eta1.real)
    # D1=np.ones_like(D1)
    # return D2
    return spla.LinearOperator((len(sec[2]),len(sec[2])),lambda v:tk.apply_F_dual(sec,T,D1,D2,v),lambda v:tk.apply_F_dual_ad(sec,T,D1,D2,v))
h=0.3
Jl=0.3
J=np.pi/4-0.1
g=np.pi/4
T=3
Fc=get_ladderF_dense(T,Jl,J,g,h)
vec=np.ones(Fc.shape[0])
vec=Fc@vec
vec=Fc@vec
vec=Fc@vec
vec=Fc@vec
# t=_
# m=t.reshape(4,(2**(2*T-2)),4,(2**(2*T-2)))[0,:,0,:]
# m
# # np.isclose(np.diag(np.diag(m)),m).all()
# np.nonzero(m>0.4)
# bin(36)
# m
# m-np.diag(np.diag(m))[1]
# m
