import numpy as np
def open_boundary_im(t):
    return np.ones((2**(2*t)))
def perfect_dephaser_im(t):
    ret=np.zeros((2**(2*t)))
    L=1<<(2*t-1)
    R=1<<(t-1)
    for i in range(2**(t-1)):
        ir=int(bin(i+2**(t-1))[3:][::-1],2)
        ret[L|(i<<t)|R|ir]=1
        ret[(i<<t)|R|ir]=1
        ret[L|(i<<t)|ir]=1
        ret[(i<<t)|ir]=1
    return ret
def dephaser_im(t,gamma):
    ret=np.zeros((2**(2*t)))
    for i in range(2**(2*t)):
        bstr=bin(i+2**(2*t))[3:]
        ret[i]=(1-gamma)**sum([a!=b for a,b in zip(bstr[1:t],bstr[::-1][:t-1])])
    return ret
