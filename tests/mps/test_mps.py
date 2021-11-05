import pytest
import numpy as np
import h5py
import imcode.mps as mps
import imcode.dense as dense
#
#
def test_store_mpo(tmpdir,seed_rng):
    f=h5py.File(tmpdir.join("test_mpo.h5"),"w")
    t,J,g,h=4,np.random.normal(),np.random.normal(),np.random.normal()
    mpo=mps.ising.ising_T(t,J,g,h)
    mpo.save_to_hdf5(f,"mpo")
    f.close()
    del mpo
    del f
    f2=h5py.File(tmpdir.join("test_mpo.h5"))
    mpo2=mps.MPO.load_from_hdf5(f2,"mpo")
    assert mpo2.to_dense()==pytest.approx(dense.ising.ising_T(t,J,g,h))
def test_store_mps(tmpdir,seed_rng):
    f=h5py.File(tmpdir.join("test_mps.h5"),"w")
    t,J,g,h=4,np.random.normal(),np.random.normal(),np.random.normal()
    mpo=mps.ising.ising_T(t,J,g,h)
    mps1=list(mps.ising.im_rectangle(mpo,chi=64))[-1]
    mps1.save_to_hdf5(f,"mps")
    f.close()
    del mps1
    del f
    f2=h5py.File(tmpdir.join("test_mps.h5"))
    mps2=mps.MPS.load_from_hdf5(f2,"mps")
    assert mps2.to_dense()==pytest.approx(dense.ising.im_diag(dense.ising.ising_T(t,J,g,h)))
def test_contract_order(seed_rng):
    L=4
    chi_mpo=5
    chi_max=64
    chi_mps=11
    dim=3
    Ws=[np.random.normal(size=(chi_mpo,chi_mpo,dim,dim))+1.0j*np.random.normal(size=(chi_mpo,chi_mpo,dim,dim)) for _ in range(L)]
    Ws[0]=Ws[0][0,:,:,:][None,:,:,:]
    Ws[-1]=Ws[-1][:,0,:,:][:,None,:,:]
    Bs=[np.random.normal(size=(chi_mps,chi_mps,dim))+1.0j*np.random.normal(size=(chi_mps,chi_mps,dim)) for _ in range(L)]
    Bs[0]=Bs[0][0,:,:][None,:,:]
    Bs[-1]=Bs[-1][:,0,:][:,None,:]
    F=mps.MPO.from_matrices(Ws)
    M=mps.MPS.from_matrices(Bs)
    Fd=F.to_dense()
    Md=M.to_dense()
    dres=Md@(Fd@(Fd@Md))
    iter_c=(F@M.copy()).contract()
    iter_c=(F@iter_c).contract(chi_max=chi_max)
    iter_c=M@iter_c
    assert iter_c==pytest.approx(dres)

    iter_c2=(F@M.copy()).contract(chi_max=chi_max)
    iter_c2=(F@iter_c2).contract(chi_max=chi_max)
    iter_c2=M@iter_c2
    assert iter_c2==pytest.approx(dres)
    # assert M.copy()@F@F@M.copy()==pytest.approx(dres)
    assert M.copy()@(F@(F@M.copy()).contract())==pytest.approx(dres)
    assert M.copy()@((F@F).contract()@M.copy())==pytest.approx(dres)
    # assert (M.copy()@F).contract()@(F@M.copy()).contract()==pytest.approx(dres)
def test_mul(seed_rng):
    L=5
    dim=4
    chi_mps=32
    Bs1=[np.random.normal(size=(chi_mps,chi_mps,dim))+1.0j*np.random.normal(size=(chi_mps,chi_mps,dim)) for _ in range(L)]
    Bs1[0]=Bs1[0][0,:,:][None,:,:]
    Bs1[-1]=Bs1[-1][:,0,:][:,None,:]
    mps1=mps.MPS.from_matrices(Bs1)
    Bs1c=[mps1.get_B(i) for i in range(L)]
    mps1c=mps.MPS.from_matrices(Bs1c,norm=mps1.tpmps.norm)
    assert mps1.to_dense()==pytest.approx(mps1c.to_dense())
    Bs2=[np.random.normal(size=(chi_mps,chi_mps,dim))+1.0j*np.random.normal(size=(chi_mps,chi_mps,dim)) for _ in range(L)]

    Bs2[0]=Bs2[0][0,:,:][None,:,:]
    Bs2[-1]=Bs2[-1][:,0,:][:,None,:]
    mps2=mps.MPS.from_matrices(Bs2)
    mpsmul=mps1*mps2
    assert mpsmul.to_dense() == pytest.approx(mps1.to_dense()*mps2.to_dense())
