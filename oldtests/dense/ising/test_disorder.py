import numpy as np
import imcode.dense as dense
import pytest
def test_dense_hr_rweights(seed_rng):
    t=3
    samples=10
    hs=np.random.normal(size=samples)#+1.0j*np.random.normal(size=samples)
    ks=[0]+sum((list(x) for x in zip(range(1,t+1),range(-1,-t-1,-1))),[])
    weights=[np.mean([np.exp(2.0j*h*k) for h in hs]) for k in ks]
    dhr=dense.ising.hr_operator(t,weights)
    dcm=np.zeros_like(dhr)
    for h in hs:
        dcm+=dense.ising.ising_h(t,h)/samples
    assert dhr==pytest.approx(dcm)

def test_dense_Jr_rweights(seed_rng):
    t=3
    tmax=10
    samples=20
    Js=np.random.normal(size=samples)#+1.0j*np.random.normal(size=samples)
    ks=[0]+sum((list(x) for x in zip(range(1,tmax+1),range(-1,-tmax-1,-1))),[])
    weights=[np.mean([np.exp(2.0j*j*k) for j in Js]) for k in ks]
    dJr=dense.ising.Jr_operator(t,weights)
    dcm=np.zeros_like(dJr)
    for j in Js:
        dcm+=dense.ising.ising_J(t,j)/samples
    assert dJr==pytest.approx(dcm)

def test_dense_Jr_fixed_sample(seed_rng):
    t=2
    samples=1000
    dJr1=dense.ising.Jr_operator(t)
    dJr2=dense.ising.Jr_operator(t,weights=[1.0]+[0.0,0.0]*t)
    assert dJr1==pytest.approx(dJr2)
    dcm=np.zeros_like(dJr1,dtype=complex)
    for _ in range(samples):
        dcm+=dense.ising.ising_J(t,np.random.uniform(0,2*np.pi))/samples
    assert dJr1==pytest.approx(dcm,abs=10/np.sqrt(samples),rel=10/np.sqrt(samples))

def test_dense_hr_fixed_sample(seed_rng):
    t=2
    samples=1000
    dhr1=dense.ising.hr_operator(t)
    dhr2=dense.ising.hr_operator(t,weights=[1.0]+[0.0,0.0]*t)
    assert dhr1==pytest.approx(dhr2)
    dcm=np.zeros_like(dhr1,dtype=complex)
    for _ in range(samples):
        dcm+=dense.ising.ising_h(t,np.random.uniform(0,2*np.pi))/samples
    assert dhr1==pytest.approx(dcm,abs=10/np.sqrt(samples),rel=10/np.sqrt(samples))

def test_dense_hr_no():
    t=3
    dhr=dense.ising.hr_operator(t,weights=np.ones((2*t+1,)))
    dcm=dense.ising.ising_h(t,0.0)
    assert dhr==pytest.approx(dcm)

def test_dense_Jr_no():
    t=3
    dJr=dense.ising.Jr_operator(t,weights=np.ones((2*t+1,)))
    dcm=dense.ising.ising_J(t,0.0)
    assert dJr==pytest.approx(dcm)
