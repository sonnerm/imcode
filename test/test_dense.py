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
    assert dense.sx(2,0)@dense.sy(2,0)==dense.sy()

    #Operators on two different sites commute



def test_ising_H():
    #Build operator manually from basic operators and compare also for complex coefficients <=> non-hermitian operators
    
    #Real coefficients <=> Hermitian operator
    pass


def test_ising_F_simple():
    def simple_ising_F(): # simple implementation of ising_F, to be compared to future implementations
        pass
    pass
def test_ising_T_obs():
    pass
