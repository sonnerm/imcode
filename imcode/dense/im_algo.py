import numpy.linalg as la
import numpy as np
def im_diag(T):
    ev,evv=la.eig(T)
    oev=evv[:,np.argmax(np.abs(ev))]
    oev/=oev[0]
    return oev
