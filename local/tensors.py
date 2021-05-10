import numpy as np


def typecheck(M, W, Wp, Wbar, u, v):
    """
    Explanation for the dimensions:
        D_D: Bond dimension of the virtual dissipative space
        D_M: Bond dimension of the new virtual memory space
        D_M_p: Bond dimension of the old virtual memory space
        d: Local qudit dimension

    Args:
        M(np.ndarray): (D_D, D_M, d, D_M_o, d)-dimensional array
        W(np.ndarray): (D_D, D_D)-dimensional array
        Wp(np.ndarray): (D_M, D_M, d, d)-dimensional array
        Wbar(np.ndarray): (D_M_o, d, d, D_M_o)-dimensional array
        u, v(np.ndarray): (d, d, d, d)-dimensional arrays

    Returns:
        bool: Always True. (If exception is found, raise it.)
    """

    Mshape, Wshape = M.shape, W.shape
    Wpshape, Wbarshape = Wp.shape, Wbar.shape
    ushape, vshape = u.shape, v.shape

    # Shape checks
    if len(Mshape)!=5:
        raise TypeError("M in the wrong shape")
    elif len(Wshape)!=2:
        raise TypeError("W in the wrong shape")
    elif len(Wpshape)!=4:
        raise TypeError("Wp in the wrong shape")
    elif len(Wbarshape)!=4:
        raise TypeError("Wbar in the wrong shape")
    elif len(u)!=4:
        raise TypeError("u in the wrong shape")
    elif len(v)!=4:
        raise TypeError("v in the wrong shape")
    
    # Internal checks
    if Mshape[2]!= Mshape[4]:
        raise TypeError("Qudit dimensions for M do not match.")

    if Wshape[0]!=Wshape[1]:
        raise TypeError("W should be a square matrix.")

    if Wpshape[0]!=Wpshape[1]:
        raise TypeError("Memory diemnsions for Wp do not match.")

    if Wpshape[2]!=Wpshape[3]:
        raise TypeError("Qudit dimensions for Wp do not match.")

    if Wbarshape[0]!=Wbarshape[3]:
        raise TypeError("Memory dimensions for Wbar do not match.")
    
    if Wbarshape[1]!=Wbarshape[2]:
        raise TypeError("Qudit dimensions for Wp do not match.")

    if len(set(ushape))>1:
        raise ValueError("Floquet unitary (u) dimension is not uniform.")

    if len(set(vshape))>1:
        raise ValueError("Floquet unitary (v) dimension is not uniform.")

    # Mutual consistency checks
    #    M(np.ndarray): (D_D, D_M, d, D_M_o, d)-dimensional array
    #    W(np.ndarray): (D_D, D_D)-dimensional array
    #    Wp(np.ndarray): (D_M, D_M, d, d)-dimensional array
    #    Wbar(np.ndarray): (D_M_o, d, d, D_M_o)-dimensional array
    #    u, v(np.ndarray): (d, d, d, d)-dimensional arrays

    # Checking M with others
    if Mshape[0]!=Wshape[0]:
        raise TypeError("M and W inconsistent")
    elif (Mshape[1]!= Wpshape[0]) or (Mshape[1]!=Wshape[1]):
        raise TypeError("M and Wp inconsistent")
    elif Mshape[2]!= Wpshape[2]:
        raise TypeError("M and Wp inconsistent")
    elif Mshape[2]!= Wbarshape[1]:
        raise TypeError("M and Wbar inconsistent")
    elif Mshape[2]!=ushape[0]:
        raise TypeError("M and u inconsistent")
    elif Mshape[2]!=vshape[0]:
        raise TypeError("M and v inconsistent")
    elif Mshape[3]!=Wbarshape[0]:
        raise TypeError("M and Wbar inconsistent")

    return True


def Mcheck(M):
    """
    Checks whether M is an isometry.

    Args:
        M(np.ndarray): (D_D, D_M, d, D_M_o, d)-dimensional array
    
    Returns:
        double: Deviation from the isometry condition.
    """
    Mshape = M.shape
    Mr = M.reshape(Mshape[0]*Mshape[1]*Mshape[2], Mshape[3]*Mshape[4])
    diff = Mr.conj().T @ Mr - np.eye(Mshape[3]*Mshape[4])
    return np.sum(abs(diff))


def Wcheck(W):
    """
    Checks whether W is an isometry.

    Args:
        W(np.ndarray): (D_D, D_D)-dimensional array
    
    Returns:
        double: Deviation from the isometry condition.
    """
    Wshape = W.shape
    diff = W.conj().T @ W - np.eye(Wshape[0])
    return np.sum(abs(diff))


def Wpcheck(Wp):
    """
    Check whether Wp is an isometry.

    Args:
        Wp(np.ndarray): (D_M, D_M, d, d)-dimensional array

    Returns:
        double: Deviation from the isometry condition.
    """
    Wpshape = Wp.shape

    Wpr = Wp.reshape(Wpshape[0], Wpshape[1] * Wpshape[2] * Wpshape[3])

    diff = Wpr @ Wpr.conj().T - np.eye(Wpshape[0])
    return np.sum(abs(diff))


def randV(n, m):
    """
    Random n x m isometry

    Args:
        n(int): Number of rows
        m(int): Number of columns

    Returns:
        np.ndarray: Isometry
    """
    X = (np.random.randn(n,m) + 1j * np.random.randn(n,m))/np.sqrt(2)
    Q, R = np.linalg.qr(X)
    R = np.diag(np.diag(R)/abs(np.diag(R)));
    return Q@R;


class Unit():
    """
    A collection of tensors for a single Floquet time step

    Explanation for the dimensions:
        D_D: Bond dimension of the virtual dissipative space
        D_M: Bond dimension of the new virtual memory space
        D_M_p: Bond dimension of the old virtual memory space
        d: Local qudit dimension

    Attributes:
        M(np.ndarray): (D_D, D_M, d, D_M_p, d)-dimensional array
        W(np.ndarray): (D_D, D_D)-dimensional array
        Wp(np.ndarray): (D_M, D_M, d, d)-dimensional array
    """
    def __init__(self, D_D, D_M, D_M_p, d):
        M = np.eye(D_D * D_M * d, D_M_p * d)
        self.M = M.reshape((D_D, D_M, d, D_M_p, d))
        self.W = np.eye(D_D, D_D)
        Wp = np.eye(D_M, D_M*d*d)
        self.Wp = Wp.reshape((D_M, D_M, d, d))
