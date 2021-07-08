from .. import MPO
import numpy as np
def im_channel_dense(im,t):
    '''
        Returns a dense operator representing the channel gauge is such that trace of environment is summation
    '''
    # postval=np.array([im.tpmps.norm])
    Bs=[]
    postval=np.array([1.0])
    for i in range(im.L-1,t,-1):
        postval=np.einsum("b,abc,c->a",postval,im.get_B(i),[0.5,0.0,0.0,0.5])
    B=im.get_B(t)
    preval=np.einsum("b,abc,c->a",postval,B,[0.5,0.0,0.0,0.5])
    ret=np.einsum("a,b,abc,cd->bcad",1/preval,postval,B,np.eye(4))
    return ret.reshape((ret.shape[0]*ret.shape[1],ret.shape[2]*ret.shape[3]))
