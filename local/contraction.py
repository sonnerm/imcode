import numpy as np
from tensors import typecheck, Mcheck, Wcheck, Wpcheck

eps=10**-6

def contract(M, W, Wp, Wbar, u, v, check=True):
    """
    Return the contraction value of the self-consistency equation.

    Args:
        Everything but the last: Tensors
        check(bool): Check whether the tensors are correct or not.

    Returns:
        float: Contraction value.
    """
    if check:
        typecheck(M, W, Wp, Wbar, u, v)
        if Mcheck(M) > eps:
            raise TypeError("M is far from being an isometry.")
        elif Wcheck(W) > eps:
            raise TypeError("W is far from being an isometry.")
        elif Wpcheck(Wp) > eps:
            raise TypeError("Wp is far from being an isometry.")

    MW = np.einsum("ijklm, kn -> ijnlm", M.conj().T, W)
    MWWp = np.einsum("ijnlm, lopw -> ijnmopw", MW, Wp)
    MWWpM = np.einsum("ijmnopw, nopxy -> ijmwxy", MWWp, M)
    MWWpMu = np.einsum("ijmwxy, jmwz -> ixyz", MWWpM, u)
    MWWpMuv = np.einsum("ixyz, yzab -> ixab", MWWpMu, v)
    final = np.einsum("ixab, xabi ->", MWWpMuv, Wbar)

    return final


def Compute_Wbar(M, W, Wp, Wbar_past, u, v, check=True):
    """
    Compute the new Wbar from the tensors at the previous step.

    Args:
        Everything but the last: Tensors
        check(bool): Check whether the tensors are correct or not.

    Returns:
        np.ndarray: Wbar
    """
    if check:
        typecheck(M, W, Wp, Wbar, u, v)
        if Mcheck(M) > eps:
            raise TypeError("M is far from being an isometry.")
        elif Wcheck(W) > eps:
            raise TypeError("W is far from being an isometry.")
        elif Wpcheck(Wp) > eps:
            raise TypeError("Wp is far from being an isometry.")

    WM = np.einsum("ij, jklop -> iklop", W, M)
    WMv = np.einsum("iklop, pwyz -> iklowyz", WM, v)
    WMuv = np.einsum("mnwx, iklowyz -> iklmnoyzx", u, WMv)
    WMuvWb = np.einsum("iklmnoyzx, oyza -> iklmnxa", WMuv, Wb)
    WMuvWbM = np.einsum("iklmnxa, axibn -> klmb", WMuvWb, M.conj().T)

    return WMuvWbM
    

    
