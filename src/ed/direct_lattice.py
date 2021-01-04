import scipy.sparse as sp
import functools
import numpy as np

def dense_kron(Ms):
    return functools.reduce(np.kron,Ms)

def dense_outer(Ms):
    return np.ravel(functools.reduce(np.outer,Ms))

def get_dense_floquet(x,y,Jx,Jy,g,h):
    U1d=np.zeros((2**(x*y)))
    szd=np.array([1,-1])
    idd=np.array([1,1])

    for xi in range(x):
        for yi in range(y):
            m=[[idd for _ in range(y)] for _ in range(x)]
            m[xi][yi]=szd
            m[(xi+1)%x][yi]=szd
            U1d+=Jx*dense_outer(functools.reduce(lambda x,y:x+y,m))
            m=[[idd for _ in range(y)] for _ in range(x)]
            m[xi][yi]=szd
            m[xi][(yi+1)%y]=szd
            U1d+=Jy*dense_kron(functools.reduce(lambda x,y:x+y,m))
            m=[[idd for _ in range(y)] for _ in range(x)]
            m[xi][yi]=szd
            U1d+=h*dense_outer(functools.reduce(lambda x,y:x+y,m))
    U1=np.diag(np.exp(1.0j*U1d))
    U2=np.eye(2**(x*y))
    sx=np.array([[np.cos(g),1.0j*np.sin(g)],[1.0j*np.sin(g),np.cos(g)]])
    id=np.array([[1.0,0.0],[0.0,1.0]])
    m=[[sx for _ in range(y)] for _ in range(x)]
    U2=dense_kron(functools.reduce(lambda x,y:x+y,m))
    return U1@U2
def get_dense_zz_diag(x,y,i,j):
    sxd=np.array([1,-1])
    idd=np.array([1,1])
    m=[[idd for _ in range(y)] for _ in range(x)]
    m[i][j]=sxd
    return dense_outer(functools.reduce(lambda x,y:x+y,m))

def get_dense_xx(x,y,i,j):
    sx=np.array([[0,1],[1,0]])
    id=np.array([[1,0],[0,1]])
    m=[[id for _ in range(y)] for _ in range(x)]
    m[i][j]=sx
    return dense_kron(functools.reduce(lambda x,y:x+y,m))

def get_dense_yy(x,y,i,j):
    sx=np.array([[0,1.0j],[-1.0j,0]])
    id=np.array([[1,0],[0,1]])
    m=[[id for _ in range(y)] for _ in range(x)]
    m[i][j]=sx
    return dense_kron(functools.reduce(lambda x,y:x+y,m))
x=9
y=1
Jx=1*np.pi/4
# Jx=0.3
Jy=Jx
g=np.pi/4
h=0.4
U=get_dense_floquet(x,y,Jx,Jy,g,h)

import transfer_dual_keldysh as tk
# Ul=tk.get_imbrie_F_p(np.array([-4*h]*x),np.array([-4*g]*x),np.array([-8*Jx]*x),x)
# Ul
#
# U
# Ul
czz_a=[[get_dense_zz_diag(x,y,i,j) for i in range(x)] for j in range(y)]
cxx_a=[[get_dense_xx(x,y,i,j) for i in range(x)] for j in range(y)]
cyy_a=[[get_dense_yy(x,y,i,j) for i in range(x)] for j in range(y)]

T=1
czzs=[]
cxxs=[]
cyys=[]
czxs=[]
cxys=[]
cyzs=[]
cxzs=[]
cyxs=[]
czys=[]
for T in range(14):
    Ut=functools.reduce(lambda x,y:x@y,[U]*T,np.eye(U.shape[0]))
    czz_c1=[[np.trace(Ut@np.diag(czz_a[0][0])@Ut.T.conj()@np.diag(czz_a[i][j]))/np.trace(U@U.T.conj()) for i in range(y)] for j in range(x)]
    cxx_c1=[[np.trace(Ut@cxx_a[0][0]@Ut.T.conj()@cxx_a[i][j])/np.trace(U@U.T.conj()) for i in range(y)] for j in range(x)]
    cyy_c1=[[np.trace(Ut@cyy_a[0][0]@Ut.T.conj()@cyy_a[i][j])/np.trace(U@U.T.conj()) for i in range(y)] for j in range(x)]
    czx_c1=[[np.trace(Ut@np.diag(czz_a[0][0])@Ut.T.conj()@cxx_a[i][j])/np.trace(U@U.T.conj()) for i in range(y)] for j in range(x)]
    cyz_c1=[[np.trace(Ut@cyy_a[0][0]@Ut.T.conj()@np.diag(czz_a[i][j]))/np.trace(U@U.T.conj()) for i in range(y)] for j in range(x)]
    cxy_c1=[[np.trace(Ut@cxx_a[0][0]@Ut.T.conj()@cyy_a[i][j])/np.trace(U@U.T.conj()) for i in range(y)] for j in range(x)]
    czy_c1=[[np.trace(Ut@np.diag(czz_a[0][0])@Ut.T.conj()@cyy_a[i][j])/np.trace(U@U.T.conj()) for i in range(y)] for j in range(x)]
    cxz_c1=[[np.trace(Ut@cxx_a[0][0]@Ut.T.conj()@np.diag(czz_a[i][j]))/np.trace(U@U.T.conj()) for i in range(y)] for j in range(x)]
    cyx_c1=[[np.trace(Ut@cyy_a[0][0]@Ut.T.conj()@cxx_a[i][j])/np.trace(U@U.T.conj()) for i in range(y)] for j in range(x)]

    czzs.append(czz_c1)
    cxxs.append(cxx_c1)
    cyys.append(cyy_c1)
    czxs.append(czx_c1)
    cyzs.append(cyz_c1)
    cxys.append(cxy_c1)
    cxzs.append(cxz_c1)
    czys.append(czy_c1)
    cyxs.append(cyx_c1)
import matplotlib.pyplot as plt

plt.imshow(np.log(np.abs(np.array(cxys).reshape(14,9))))

plt.imshow(np.log(np.abs(np.array(czzs).reshape(14,11))))
plt.colorbar()
    # print("zz")
    # print(np.array(czz_c1).real)
    # print("xx")
    # print(np.array(cxx_c1).real)
    # print("xz")
    # print(np.array(czx_c1).real)

    # U2=np.zeros_like(U1)
    # esx=np.expm
    # for xi in range(x):
    #     for yi in range(y):
    #         m=[[id for _ in range(x)] for _ in range(y)]
    #         m[xi][yi]=sz
    #         m[(xi+1)%x][yi]=sz
    #         U1+=Jx*dense_kron(functools.reduce(lambda x,y:x+y,m))
    #         m=[[id for _ in range(x)] for _ in range(y)]
    #         m[xi][yi]=sz
    #         m[xi][(yi+1)%y]=sz
    #         U1+=Jy*dense_kron(functools.reduce(lambda x,y:x+y,m))
    #         m=[[id for _ in range(x)] for _ in range(y)]
    #         m[xi][yi]=sz
    #         U1+=h*dense_kron(functools.reduce(lambda x,y:x+y,m))
