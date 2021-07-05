import numpy as np
def unitary_channel(F):
    return np.kron(F,F.conj())
