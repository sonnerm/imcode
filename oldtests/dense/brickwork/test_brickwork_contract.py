import imcode.dense as dense
import numpy as np
import numpy.linalg as la
from functools import reduce
import pytest
def test_contract_2x3_mixed(seed_rng):
    L=2
    t=3
    init=np.random.normal(size=(4,4))+1.0j*np.random.normal(size=(4,4))
    final=np.random.normal(size=(4,4))+1.0j*np.random.normal(size=(4,4))
    gate=np.random.normal(size=(4,4))+1.0j*np.random.normal(size=(4,4))
    F=dense.brickwork.brickwork_F(L,[gate]*(L-1))
    acc=init
    for _ in range(t):
        acc=F@acc@F.T.conj()
    direct=np.trace(final@acc)
    bc=dense.brickwork.brickwork_La(t)
    transverse=bc@dense.brickwork.brickwork_Sb(t,dense.unitary_channel(gate),init,final)@bc
    assert transverse==pytest.approx(direct)

def test_contract_2x3_separable(seed_rng):
    L=2
    t=3
    init=[np.random.normal(size=(2,2))+1.0j*np.random.normal(size=(2,2)) for _ in range(L)]
    final=[np.random.normal(size=(2,2))+1.0j*np.random.normal(size=(2,2)) for _ in range(L)]
    gate=[np.random.normal(size=(2,2))+1.0j*np.random.normal(size=(2,2)) for _ in range(L)]
    F=dense.brickwork.brickwork_F(L,[dense.kron(gate)])
    acc=np.kron(init[0],init[1])
    for _ in range(t):
        acc=F@acc@F.T.conj()
    direct=np.trace(np.kron(final[0],final[1])@acc)
    bcl=dense.brickwork.brickwork_Lb(t,dense.unitary_channel(gate[1]),init[1],final[1])
    bcr=dense.brickwork.brickwork_Lb(t,dense.unitary_channel(gate[0]),init[0],final[0])
    transverse1=bcr@dense.brickwork.brickwork_Sa(t,dense.unitary_channel(np.eye(4)))@bcl
    assert transverse1==pytest.approx(direct)
    bc=dense.brickwork.brickwork_La(t)
    transverse=bc@dense.brickwork.brickwork_Sb(t,dense.unitary_channel(dense.kron(gate)),np.kron(init[0],init[1]),np.kron(final[0],final[1]))@bc
    assert transverse==pytest.approx(direct)

#
def test_contract_3x3(seed_rng):
    L=3
    t=3
    init=[np.random.normal(size=(2,2))+1.0j*np.random.normal(size=(2,2)) for _ in range(L)]
    final=[np.random.normal(size=(2,2))+1.0j*np.random.normal(size=(2,2)) for _ in range(L)]
    gate=[np.random.normal(size=(4,4))+1.0j*np.random.normal(size=(4,4)) for _ in range(L-1)]
    F=dense.brickwork.brickwork_F(L,gate)
    acc=dense.kron(init)
    finalvec=dense.kron(final)
    for _ in range(t):
        acc=F@acc@F.T.conj()
    direct=np.trace(finalvec@acc)
    bcla=dense.brickwork.brickwork_La(t)
    bclb=dense.brickwork.brickwork_Lb(t,dense.unitary_channel(np.eye(2)),init[2],final[2])
    sa=dense.brickwork.brickwork_Sa(t,dense.unitary_channel(gate[1]))
    sb=dense.brickwork.brickwork_Sb(t,dense.unitary_channel(gate[0]),np.kron(init[0],init[1]),np.kron(final[0],final[1]))
    transverse=bcla@(sb@(sa@bclb))
    assert transverse==pytest.approx(direct)
def test_contract_3x3_reversed(seed_rng):
    L=3
    t=3
    init=[np.random.normal(size=(2,2))+1.0j*np.random.normal(size=(2,2)) for _ in range(L)]
    final=[np.random.normal(size=(2,2))+1.0j*np.random.normal(size=(2,2)) for _ in range(L)]
    gate=[np.random.normal(size=(4,4))+1.0j*np.random.normal(size=(4,4)) for _ in range(L-1)]
    F=dense.brickwork.brickwork_F(L,gate,reversed=True)
    acc=dense.kron(init)
    finalvec=dense.kron(final)
    for _ in range(t):
        acc=F@acc@F.T.conj()
    direct=np.trace(finalvec@acc)
    bcla=dense.brickwork.brickwork_La(t)
    bclb=dense.brickwork.brickwork_Lb(t,dense.unitary_channel(np.eye(2)),init[0],final[0])
    sa=dense.brickwork.brickwork_Sa(t,dense.unitary_channel(gate[0]))
    sb=dense.brickwork.brickwork_Sb(t,dense.unitary_channel(gate[1]),np.kron(init[1],init[2]),np.kron(final[1],final[2]))
    transverse=bclb@(sa@(sb@bcla))
    assert transverse==pytest.approx(direct)

def test_contract_4x3(seed_rng):
    L=4
    t=3
    init=[np.random.normal(size=(4,4))+1.0j*np.random.normal(size=(4,4)) for _ in range(L//2)]
    final=[np.random.normal(size=(4,4))+1.0j*np.random.normal(size=(4,4)) for _ in range(L//2)]
    gate=[np.random.normal(size=(4,4))+1.0j*np.random.normal(size=(4,4)) for _ in range(L-1)]
    F=dense.brickwork.brickwork_F(L,gate)
    acc=dense.kron(init)
    finalvec=dense.kron(final)
    for _ in range(t):
        acc=F@acc@F.T.conj()
    direct=np.trace(finalvec@acc)
    bc=dense.brickwork.brickwork_La(t)
    sa=dense.brickwork.brickwork_Sa(t,dense.unitary_channel(gate[1]))
    sb1=dense.brickwork.brickwork_Sb(t,dense.unitary_channel(gate[0]),init[0],final[0])
    sb2=dense.brickwork.brickwork_Sb(t,dense.unitary_channel(gate[2]),init[1],final[1])
    transverse=bc@(sb1@(sa@(sb2@bc)))
    assert transverse==pytest.approx(direct)

def test_contract_3x3_unity(seed_rng):
    L=3
    t=3
    init=[np.random.normal(size=(2,2))+1.0j*np.random.normal(size=(2,2)) for _ in range(L)]
    init=[i.T.conj()+i for i in init]
    init=[i/np.trace(i) for i in init]
    final=[np.eye(2) for _ in range(L)]
    gate=[np.random.normal(size=(4,4))+1.0j*np.random.normal(size=(4,4)) for _ in range(L-1)]
    gate=[g.T.conj()+g for g in gate]
    gate=[la.eigh(g)[1] for g in gate]
    bcla=dense.brickwork.brickwork_La(t)
    bclb=dense.brickwork.brickwork_Lb(t,dense.unitary_channel(np.eye(2)),init[0],final[0])
    sa=dense.brickwork.brickwork_Sa(t,dense.unitary_channel(gate[0]))
    sb=dense.brickwork.brickwork_Sb(t,dense.unitary_channel(gate[1]),np.kron(init[1],init[2]),np.kron(final[1],final[2]))
    transverse=bclb@(sa@(sb@bcla))
    assert transverse==pytest.approx(1.0)
