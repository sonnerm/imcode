import imcode.mps as mps
import numpy as np
import imcode.dense as dense
import pytest
def test_mps_hr_even_rweights(seed_rng):
    t=4
    weights=np.random.normal(size=(2*t+1,))+1.0j*np.random.normal(size=(2*t+1,))
    dhr=dense.ising.hr_operator(t,weights)
    mhr=mps.ising.hr_operator(t,weights)
    assert mhr.to_dense()==pytest.approx(dhr)


def test_mps_hr_odd_rweights(seed_rng):
    t=3
    weights=np.random.normal(size=(2*t+1,))+1.0j*np.random.normal(size=(2*t+1,))
    dhr=dense.ising.hr_operator(t,weights)
    mhr=mps.ising.hr_operator(t,weights)
    assert mhr.to_dense()==pytest.approx(dhr)

def test_mps_hr_even_fixed():
    t=4
    dhr=dense.ising.hr_operator(t)
    mhr=mps.ising.hr_operator(t)
    assert mhr.to_dense()==pytest.approx(dhr)
    dhr=dense.ising.hr_operator(t,np.ones((2*t+1,)))
    mhr=mps.ising.hr_operator(t,np.ones((2*t+1,)))
    assert mhr.to_dense()==pytest.approx(dhr)

def test_mps_hr_odd_fixed():
    t=3
    dhr=dense.ising.hr_operator(t)
    mhr=mps.ising.hr_operator(t)
    assert mhr.to_dense()==pytest.approx(dhr)
    dhr=dense.ising.hr_operator(t,np.ones((2*t+1,)))
    mhr=mps.ising.hr_operator(t,np.ones((2*t+1)))
    assert mhr.to_dense()==pytest.approx(dhr)

def test_mps_Jr_even_rweights(seed_rng):
    t=4
    weights=np.random.normal(size=(2*t+1,))+1.0j*np.random.normal(size=(2*t+1,))
    dJr=dense.ising.Jr_operator(t,weights)
    mJr=mps.ising.Jr_operator(t,weights)
    assert mJr.to_dense()==pytest.approx(dJr)


def test_mps_Jr_odd_rweights(seed_rng):
    t=3
    weights=np.random.normal(size=(2*t+1,))+1.0j*np.random.normal(size=(2*t+1,))
    dJr=dense.ising.Jr_operator(t,weights)
    mJr=mps.ising.Jr_operator(t,weights)
    assert mJr.to_dense()==pytest.approx(dJr)

def test_mps_Jr_even_fixed():
    t=4
    dJr=dense.ising.Jr_operator(t)
    mJr=mps.ising.Jr_operator(t)
    assert mJr.to_dense()==pytest.approx(dJr)
    dJr=dense.ising.Jr_operator(t,np.ones((2*t+1,)))
    mJr=mps.ising.Jr_operator(t,np.ones((2*t+1,)))
    assert mJr.to_dense()==pytest.approx(dJr)

def test_mps_Jr_odd_fixed():
    t=3
    dJr=dense.ising.Jr_operator(t)
    mJr=mps.ising.Jr_operator(t)
    assert mJr.to_dense()==pytest.approx(dJr)
    dJr=dense.ising.Jr_operator(t,np.ones((2*t+1,)))
    mJr=mps.ising.Jr_operator(t,np.ones((2*t+1)))
    assert mJr.to_dense()==pytest.approx(dJr)
