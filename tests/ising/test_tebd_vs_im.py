import imcode
import pytest
import ttarray as tt
import numpy as np
from imcode import SZ,SX,ID,ZE
import functools
def mkron(a):
    return functools.reduce(np.kron,a)

def mouter(a):
    return functools.reduce(np.outer,a).ravel()
def test_product_homhom(seed_rng):
    L=7
    t=6
    J=np.random.random()-0.5
    g=np.random.random()-0.5
    h=np.random.random()-0.5
    init=[np.random.random((2,))+np.random.random((2,))*1.0j-0.5-0.5j for _ in range(L)]
    init=[i.T.conj()+i for i in init]
    init=[i/np.sqrt(np.sum(i.conj()*i)) for i in init]
    init=[init[0] for _ in init]
    F=imcode.ising_F(L,J,g,h)
    opr2=np.random.random((4,4))-0.5
    opr2+=opr2.T.conj()
    ope=np.random.random((2,2))-0.5
    ope+=ope.T.conj()
    ope2=np.random.random((4,4))+np.random.random((4,4))*1.0j-0.5-0.5j
    ope2+=ope2.T.conj()
    # dense:
    dF=np.array(F)
    dzzl,dzzr2,dzze,dzze2=[],[],[],[]
    dopl=mkron([SZ]+[ID]*(L-1))
    dopr2=mkron([ID]*(L-2)+[opr2])
    dope=mkron([ID]*(L//2)+[ope]+[ID]*(L//2))
    dope2=mkron([ID]*(L//2)+[ope2]+[ID]*(L//2-1))
    dinit=mouter(init)
    for _ in range(t):
        dzzl.append(dinit.T.conj()@dopl@dinit)
        dzzr2.append(dinit.T.conj()@dopr2@dinit)
        dzze.append(dinit.T.conj()@dope@dinit)
        dzze2.append(dinit.T.conj()@dope2@dinit)
        dinit=dF@dinit
    dzzl.append(dinit.T.conj()@dopl@dinit)
    dzzr2.append(dinit.T.conj()@dopr2@dinit)
    dzze.append(dinit.T.conj()@dope@dinit)
    dzze2.append(dinit.T.conj()@dope2@dinit)
    #tebd wf
    # wzzl,wzzr2,wzze,wzze2=[],[],[],[]
    # wopl=tt.fromproduct([SZ]+[ID]*(L-1))
    # wopr2=tt.fromproduct([ID]*(L-2)+[opr2])
    # wope=tt.fromproduct([ID]*(L//2)+[ope]+[ID]*(L//2))
    # wope2=tt.fromproduct([ID]*(L//2)+[ope2]+[ID]*(L//2-1))
    # winit=tt.fromproduct(init)
    # for _ in range(t):
    #     wzzl.append(np.array(winit.T.conj()@wopl@winit))
    #     wzzr2.append(np.array(winit.T.conj()@wopr2@winit))
    #     wzze.append(np.array(winit.T.conj()@wope@winit))
    #     wzze2.append(np.array(winit.T.conj()@wope2@winit))
    #     winit=F@winit
    #     winit.truncate(chi_max=64)
    # wzzl.append(np.array(winit.T.conj()@wopl@winit))
    # wzzr2.append(np.array(winit.T.conj()@wopr2@winit))
    # wzze.append(np.array(winit.T.conj()@wope@winit))
    # wzze2.append(np.array(winit.T.conj()@wope2@winit))
    # assert wzzl==pytest.approx(dzzl)
    # assert wzzr2==pytest.approx(dzzr2)
    # assert wzze==pytest.approx(dzze)
    # assert wzze2==pytest.approx(dzze2)
    #
    # #tebd dm
    # mzzl,mzzr2,mzze,mzze2=[],[],[],[]
    # minit=tt.fromproduct([np.outer(i.T.conj(),i) for i in init])
    # for _ in range(t):
    #     mzzl.append(np.array(np.trace(wopl@minit)))
    #     mzzr2.append(np.array(np.trace(wopr2@minit)))
    #     mzze.append(np.array(np.trace(wope@minit)))
    #     mzze2.append(np.array(np.trace(wope2@minit)))
    #     minit=(F@minit@F.T.conj())
    #     minit.truncate(chi_max=64)
    #
    # mzzl.append(np.array(np.trace(wopl@minit)))
    # mzzr2.append(np.array(np.trace(wopr2@minit)))
    # mzze.append(np.array(np.trace(wope@minit)))
    # mzze2.append(np.array(np.trace(wope2@minit)))
    #
    # assert mzzl==pytest.approx(dzzl)
    # assert mzzr2==pytest.approx(dzzr2)
    # assert mzze==pytest.approx(dzze)
    # assert mzze2==pytest.approx(dzze2)

    #im rectangle
    T=imcode.ising_T(t,J,g,h)
    Ts=[T for _ in range(t)]
    rinitl=tt.fromproduct([np.outer(i.T.conj(),i) for i in init])
    rinitr=tt.fromproduct([np.outer(i.T.conj(),i) for i in init[::-1]])
    riml=list(imcode.zoz_lcga(Ts,rinitl,chi_max=64))
    rimr=list(imcode.zoz_lcga(Ts,rinitr,chi_max=64))
    ch1=np.array(imcode.unitary_channel(imcode.ising_F(1,J,g,h)))
    ch2=np.array(imcode.unitary_channel(imcode.ising_F(2,J,g,h)))
    rzzl=imcode.ising_boundary_evolution(rimr[-2],ch1,init=np.outer(init[0].T.conj(),init[0]))
    # rzzr2=imcode.ising_boundary_evolution(riml[-2],ch2,init=tt.fromproduct([np.outer(init[-1],init[-1].conj(),np.outer(init[-2],init[-2]).conj()]))
    # rzze=imcode.ising_embedded_evolution(riml[L//2],ch1,riml[L//2-1],init=init[L//2])
    # rzze2=imcode.ising_embedded_evolution(riml[L//2],ch2,riml[L//2-2],init=tt.fromproduct([init[L//2],init[L//2+1]]).toslice())
    assert [np.trace(SZ@np.array(imcode.unvectorize_operator(x))) for x in rzzl]==pytest.approx(dzzl)
    # assert list(rzzr2)==pytest.approx(dzzr2)
    # assert list(rzze)==pytest.approx(dzze)
    # assert list(rzze2)==pytest.approx(dzze2)

    #im lcga
    # Ts=[imcode.ising_T(t,J,g,h) for t in range(1,t+1)]
    # linitl=tt.fromproduct([np.outer(i.T.conj(),i) for i in init])
    # linitr=tt.fromproduct([np.outer(i.T.conj(),i) for i in init[::-1]])
    # liml=list(imcode.zoz_lcga(Ts,linitl,chi_max=64))
    # liml=list(imcode.zoz_lcga(Ts,linitr,chi_max=64))
    # lzzl=imcode.ising_boundary_evolution(riml[-1],imcode.ising_F(1,J,g,h),init[0])
    # lzzr2=imcode.ising_boundary_evolution()
    # lzze=imcode.ising_embedded_evolution()
    # lzze2=imcode.ising_embedded_evolution()








def test_product_hethom():
    pass

def test_product_homhet():
    pass

def test_product_hethet():
    pass

def test_mps_homhom():
    pass

def test_mps_hethom():
    pass

def test_mps_homhet():
    pass

def test_mps_hethet():
    pass
