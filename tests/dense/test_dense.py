import pytest

import imcode.dense as dense
import numpy as np
import scipy.linalg as scla
#
#
def test_basic_ops():
    # Are one site many body operators sane?
    assert (dense.sx(1, 0) == dense.SX).all()
    assert (dense.sy(1, 0) == dense.SY).all()
    assert (dense.sz(1, 0) == dense.SZ).all()
    assert (dense.sp(1, 0) == dense.SP).all()
    assert (dense.sm(1, 0) == dense.SM).all()

    # Testing algebra on both sites in a 2 site system
    assert (dense.sx(2, 0) @ dense.sx(2, 0) == dense.one(2)).all()
    assert (dense.sy(2, 0) @ dense.sy(2, 0) == dense.one(2)).all()
    assert (dense.sz(2, 0) @ dense.sz(2, 0) == dense.one(2)).all()

    assert (dense.sx(2, 1) @ dense.sx(2, 1) == dense.one(2)).all()
    assert (dense.sy(2, 1) @ dense.sy(2, 1) == dense.one(2)).all()
    assert (dense.sz(2, 1) @ dense.sz(2, 1) == dense.one(2)).all()

    assert (dense.sx(2, 1) @ dense.sy(2, 1) == 1.0j * dense.sz(2, 1)).all()
    assert (dense.sx(2, 1) @ dense.sz(2, 1) == -1.0j * dense.sy(2, 1)).all()

    # Testing addition
    lhs = (dense.sx(2, 0) + dense.sz(2, 1)) @ (dense.sy(2, 0) + dense.sy(2, 1))
    rhs = 1.0j * dense.sz(2, 0) + dense.sx(2, 0) @ dense.sy(2, 1) + \
        dense.sz(2, 1) @ dense.sy(2, 0) - 1.0j * dense.sx(2, 1)
    assert (lhs == rhs).all()
    # Operators on different sites commute
    assert (dense.sx(2, 1) @ dense.sz(2, 0) ==
            dense.sz(2, 0) @ dense.sx(2, 1)).all()
    assert ((dense.sx(3, 1) + dense.sy(3, 2)) @ dense.sz(3, 0) ==
            dense.sz(3, 0) @ (dense.sx(3, 1) + dense.sy(3, 2))).all()
