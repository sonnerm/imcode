import numpy as np
from ..dense import open_boundary_im
def boundary_obs(im,obs):
    return np.sum(im*obs)/2
def embedded_obs(left_im,obs_op,right_im):
    return np.sum(left_im*(obs_op@right_im))/2
def zz_state(t):
    ret=np.zeros((2,2**(t-1),2,2**(t-1)))
    ret[0,:,0,:]=1
    ret[0,:,1,:]=-1
    ret[1,:,0,:]=-1
    ret[1,:,1,:]=1
    return np.ravel(ret)
def embedded_czz(im,lop):
    t=int(np.log2(im.shape[0]))//2
    return embedded_obs(im,lop,zz_state(t)*im)
def boundary_czz(im,lop):
    t=int(np.log2(im.shape[0]))//2
    st=lop@zz_state(t)
    return boundary_obs(im,st)
def embedded_norm(im,lop):
    return embedded_obs(im,lop,im)
def boundary_norm(im,lop):
    t=int(np.log2(im.shape[0]))//2
    st=lop@open_boundary_im(t)
    return boundary_obs(im,st)

def direct_czz(F,t,i,j):
    return None
    # L=int(np.log2(F.shape[0]))
    # return np.trace(la.matrix_power(F,t)@sz(L,i)@la.matrix_power(F.T.conj(),t)@sz(L,j))/(2**L)

def direct_norm(F,t,i,j): #lol
    return None
    # L=int(np.log2(F.shape[0]))
    # return np.trace(la.matrix_power(F,t)@la.matrix_power(F.T.conj(),t))/(2**L)
