from imcode import sparse
import pytest
import numpy as np
import hashlib
def seed_rng(stri):
    np.random.seed(int.from_bytes(hashlib.md5(stri.encode('utf-8')).digest(),"big")%2**32)
