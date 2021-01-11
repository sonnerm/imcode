import pytest
import numpy as np
import tenpy.tools.hdf5_io as hdf5_io
import h5py
import imcode.mps as mps
import imcode.dense as dense
from .utils import seed_rng

def test_store_mpo(tmpdir):
    f=h5py.File(tmpdir.join("test_mpo.h5"),"w")
    seed_rng("store_mpo")
    t,J,g,h=4,np.random.normal(),np.random.normal(),np.random.normal()
    mpo=mps.ising_T(t,J,g,h)
    hdf5_io.save_to_hdf5(f,mpo,"mpo")
    f.close()
    del mpo
    del f
    f2=h5py.File(tmpdir.join("test_mpo.h5"))
    mpo2=hdf5_io.load_from_hdf5(f2,"mpo")
    assert mps.mpo_to_dense(mpo2)==pytest.approx(dense.ising_T(t,J,g,h))
def test_store_mps(tmpdir):
    f=h5py.File(tmpdir.join("test_mps.h5"),"w")
    seed_rng("store_mps")
    t,J,g,h=4,np.random.normal(),np.random.normal(),np.random.normal()
    mpo=mps.ising_T(t,J,g,h)
    mps1=mps.im_iterative(mpo,chi=128)
    hdf5_io.save_to_hdf5(f,mps1,"mps")
    f.close()
    del mps1
    del f
    f2=h5py.File(tmpdir.join("test_mps.h5"))
    mps2=hdf5_io.load_from_hdf5(f2,"mps")
    assert mps.mps_to_dense(mps2)==pytest.approx(dense.im_iterative(dense.ising_T(t,J,g,h)))
