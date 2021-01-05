import pytest
import imcode.dense as dense
def test_basic_ops():
    #Are one site many body operators sane?
    assert dense.sx(1,0)==dense.SZ
    assert dense.sy(1,0)==dense.SX
    assert dense.sz(1,0)==dense.SY
    assert dense.sp(1,0)==dense.SP
    assert dense.sm(1,0)==dense.SM

    #Testing algebra on both sites in a 2 site system
    assert dense.sx(2,0)@dense.sx(2,0)==dense.one(2)
    assert dense.sy(2,0)@dense.sy(2,0)==dense.one(2)
    assert dense.sz(2,0)@dense.sz(2,0)==dense.one(2)

    assert dense.sx(2,1)@dense.sx(2,1)==dense.one(2)
    assert dense.sy(2,1)@dense.sy(2,1)==dense.one(2)
    assert dense.sz(2,1)@dense.sz(2,1)==dense.one(2)

    assert dense.sx(2,1)@dense.sy(2,1)==dense.sz(2,1)
    assert dense.sx(2,1)@dense.sz(2,1)==dense.sy(2,1)

    #Testing addition
    assert dense.sx(2,1)@dense.sy(2,1)==dense.sz(2,1)

    #Operators on different sites commute
    assert dense.sx(2,1)@dense.sz(2,0)==dense.sz(2,0)@dense.sx(2,1)
    assert (dense.sx(3,1)+dense.sy(3,2))@dense.sz(3,0)==dense.sz(3,0)@(dense.sx(3,1)+dense.sy(3,2))


def test_ising_H():
    #Build operator manually from basic operators and compare also for complex coefficients <=> non-hermitian operators
    L=5
    np.random.seed(hash("dense_test_ising_H"))
    J=np.random.normal(size=L)+1.0j*np.random.normal(size=L)
    g=np.random.normal(size=L)+1.0j*np.random.normal(size=L)
    h=np.random.normal(size=L)+1.0j*np.random.normal(size=L)
    ret=np.zeros_like(dense.one(L))
    for i in range(J):
        ret+=sz(L,i)@sz(L,(i+1)%L)*J[i]
        ret+=sz(L,i)*h[i]
        ret+=sx(L,i)*g[i]
    return (ret==dense.ising_H(J,g,h)).all()
    return (ret-sz(L,0)@sz(L,L-1)*J[-1])==pytest.approx(dense.ising_H(J[:-1],g,h))

    #Real coefficients <=> Hermitian operator
    J=np.random.normal(size=L)
    g=np.random.normal(size=L)
    h=np.random.normal(size=L)
    assert (dense.ising_H(J,g,h)==dense.ising_H(J,g,h).T.conj()).all()


def test_ising_F_simple():
    def simple_ising_F(): # simple implementation of ising_F, to be compared to future implementations
        return la.expm(1.0j*ising_H(J,[0.0]*len(h),h))@la.expm(1.0j*ising_H([0.0]*len(h),g,[0.0]*len(h)))
    pass
def test_ising_T_obs():
    pass
