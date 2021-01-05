import numpy as np

def fwht(a):
    '''
        Performs an inplace Fast-Walsh-Hadamard transformation on array a.
    '''
    h = 1
    slen=len(a)
    while h < slen:
        a=a.reshape((slen//h,h))
        a[::2,:],a[1::2,:]=a[::2,:]+a[1::2,:],a[::2,:]-a[1::2,:]
        a=a.reshape((slen,))
        h *= 2
