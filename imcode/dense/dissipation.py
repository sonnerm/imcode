import numpy as np
from .im import dephaser_im
from .utils import dense_kron

def dephaser_operator(t,gamma):
    return np.diag(dephaser_im(t,gamma))

def depolarizer_operator(t,gamma):
    ret= np.ones((2**(2*t),2**(2*t)))
    for i in range(t):
        fwind=(2**(1+i),2,2**(1+2*t+i),2,2**(t-i))
        ret=ret.reshape(fwind+fwind)
        ret[:,1,:,1,:,:,1,:,1,:]*=1
        ret[:,1,:,1,:,:,0,:,0,:]*=1
        ret[:,0,:,0,:,:,1,:,1,:]*=1
        ret[:,0,:,0,:,:,0,:,0,:]*=1

        ret[:,1,:,1,:,:,1,:,0,:]*=0
        ret[:,1,:,1,:,:,0,:,1,:]*=0
        ret[:,0,:,0,:,:,1,:,0,:]*=0
        ret[:,0,:,0,:,:,0,:,1,:]*=0

        ret[:,1,:,0,:,:,1,:,0,:]*=1
        ret[:,1,:,0,:,:,0,:,1,:]*=0
        ret[:,0,:,0,:,:,1,:,0,:]*=0
        ret[:,0,:,0,:,:,0,:,1,:]*=0


    # prim=np.array([[1.0-p/2,p/2,0.0,0.0],[p/2,1.0-p/2,0.0,0.0],[0.0,0.0,1.0-p,0.0],[0.0,0.0,0.0,1.0-p]])
