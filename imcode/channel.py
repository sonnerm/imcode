import numpy as np
import ttarray as tt
import math

def unitary_channel(F):
    if len(F.shape)==2:
        L=int(math.log2(F.shape[0]))
        F=tt.asarray(F,cluster=((2,2),)*L)
        Ws=[np.einsum("abcd,efgh->aebfcgdh",W,W.conj()) for W in F.tomatrices_unchecked()]
        Ws=[W.reshape((W.shape[0]**2,4,4,W.shape[-1]**2)) for W in Ws]
        return tt.frommatrices(Ws)
    if len(F.shape)==4:
        #not sure if that is necessary, but ...
        L=int(math.log2(F.shape[1]))
        F=tt.asslice(F,cluster=((2,2),)*L)
        Ws=[np.einsum("abcd,efgh->aebfcgdh",W,W.conj()) for W in F.tomatrices_unchecked()]
        Ws=[W.reshape((W.shape[0]**2,4,4,W.shape[-1]**2)) for W in Ws]
        return tt.frommatrices_slice(Ws)
    else:
        raise ValueError("unitary operator must be either 2D (array) or 4D (slice)")

# def dephasing_channel(gamma,basis=np.eye(2)):
#     D=np.diag([1.0,1.0-gamma,1.0-gamma,1.0])
#     U=unitary_channel(basis)
#     return U.T.conj()@D@U
# def depolarizing_channel(p,dm=np.eye(2)/2):
#     d=dm.shape[0]
#     ret=np.eye(d**2)*(1-p)
#     ret+=np.outer(dm.ravel(),np.eye(d).ravel())*p
#     return ret

def vectorize_operator(op):
    if len(op.shape)==2:
        L=int(math.log2(op.shape[0]))
        op=tt.asarray(op,cluster=((2,2),)*L)
        return tt.frommatrices([o.reshape((o.shape[0],4,o.shape[-1])) for o in op.tomatrices_unchecked()])
    if len(op.shape)==4:
        L=int(math.log2(op.shape[1]))
        op=tt.asslice(op,cluster=((2,2),)*L)
        return tt.frommatrices_slice([o.reshape((o.shape[0],4,o.shape[-1])) for o in op.tomatrices_unchecked()])
    else:
        raise ValueError("Operator must either be 2d (array) or 4d (slice), but has %i dimension"%(len(op.shape)))

def unvectorize_operator(state):
    if len(state.shape)==1:
        L=int(math.log2(state.shape[0]))//2
        state=tt.asarray(state,cluster=((4,),)*L)
        return tt.frommatrices([o.reshape((o.shape[0],2,2,o.shape[-1])) for o in state.tomatrices_unchecked()])
    if len(state.shape)==3:
        L=int(math.log2(state.shape[1]))//2
        state=tt.asslice(state,cluster=((4,),)*L)
        return tt.frommatrices_slice([o.reshape((o.shape[0],2,2,o.shape[-1])) for o in state.tomatrices_unchecked()])
    else:
        raise ValueError("Vectorized operator must either be 2d (array) or 4d (slice), but has %i dimension"%(len(op.shape)))
