from .. import MPO
import numpy as np
def im_channel_dense(im,t):
    '''
        Returns a dense operator representing the channel gauge is such that trace of environment is summation
    '''
    postval=im.get_S(im.L-1)
    for i in range(im.L-1,t):
        postval=np.einsum("b,abc,b,c->a",postval,im.get_B(t),im.get_S(t),[0.5,0.0,0.0,0.5])
    B=np.einsum("abc,b->abc",im.get_B(t),im.get_S(t))
    preval=np.einsum("b,abc,c->a",postval,B,[1.0,0.0,0.0,1.0])
    ret=np.einsum("a,b,abc,cd->bcad",preval,1/postval,B,np.eye(4))
    return ret.reshape((ret.shape[0]*ret.shape[1],ret.shape[2]*ret.shape[3]))
