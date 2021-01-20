import pytest
import numpy as np
import imcode.dense as dense
def check_dense_im(im):
    t=int(np.log2(len(im)))//2
    assert im[np.nonzero(dense.perfect_dephaser_im(t))]==pytest.approx(np.ones((2**(t+1))))
    #check norm boundary
    #check norm embedded
    pass

def check_dense_imp(imp):
    #check classical configurations
    #check norm boundary
    #check norm embedded
    pytest.skip("not implemented yet")
    pass

def check_mps_im(im):
    #check classical configurations
    #check norm boundary
    #check norm embedded
    pytest.skip("not implemented yet")
    pass

def check_mps_imp(imp):
    #check classical configurations
    #check norm boundary
    #check norm embedded
    pytest.skip("not implemented yet")
    pass
