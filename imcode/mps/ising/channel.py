from .. import MPO
import numpy as np
def im_channel_dense(im,t):
    '''
        Returns a dense operator representing the channel
    '''
    ret=np.einsum("abc,b,cd->bcad",im.get_B(t),im.get_S(t),np.eye(4))
    return ret.reshape((ret.shape[0]*ret.shape[1],ret.shape[2]*ret.shape[3]))
