import numpy as np
from . import MPO
def unitary_channel(F):
    Ws=F.get_Ws()
    Ws=[np.einsum("abcd,efgh->aebfcgdh",W,W.conj()).reshape((W.shape[0]**2,W.shape[1]**2,W.shape[2]**2,W.shape[3]**2)) for W in Ws]
    return MPO.from_matrices(Ws)
