import numpy as np
def _get_mat(even,odd):
    ret=np.zeros((2,2,2))
    ret[1,0,1]=odd[0]
    ret[1,1,0]=odd[1]
    ret[0,0,0]=even[0]
    ret[0,1,1]=even[1]
    return ret
_FERMI_A=_get_mat([1,1],[1,1])
_FERMI_B=_get_mat([1,-1],[1,1])
def brickwork_fermi_to_spin(im,truncate=True):
    t=int(math.log2(im.shape[0]/4))
    Omps=tt.frommatrices([_FERMI_A[0,...][None,...]]+[_FERMI_B,_FERMI_A]*(t*2-1)+[_FERMI_B[...,0][...,None]])
    cprev=im.chi
    im=im*Omps
    if truncate:
        im.truncate(chi_max=max(cprev)) #later exact replication of chi structure
    return im
