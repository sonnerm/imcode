import pytest
import imcode
import importlib.resources
import numpy as np
import h5py
from . import data
def test_hdf5_legacy_mpo(seed_rng):
    with importlib.resources.path(data,"test_mpo.hdf5") as p:
        f=h5py.File(p,"r")
    testi=imcode.loadhdf5(f,"mpo")
    assert testi.shape==(4**10,4**10)
    assert testi.cluster==((4,4),)*10
    assert testi.chi==(4,)*9

def test_hdf5_legacy_mps(seed_rng):
    with importlib.resources.path(data,"test_mps.hdf5") as p:
        f=h5py.File(p,"r")
    testi=imcode.loadhdf5(f,"mps")
