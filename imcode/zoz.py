import ttarray as tt
import numpy as np
'''
    ZOZ gate order:
        1
        |
    0 - o - 3
        |
        2
'''
# ID_ZOZ
# ID_OZ=
# ID_ZO=

def zoz_H(L,zozs):
    pass
def zoz_F(L,zozs,oz=ID_OZ,zo=ID_ZO,reversed=False):
    pass

def zoz_Fe(L,zozse,oz=ID_OZ):
    pass

def zoz_Fo(L,zozso,oz=ID_OZ,zo=ID_ZO):
    zozso=np.array(zozso)
    if len(zozso.shape)==4:
        zozso=np.array([zozso]*(L//2))



def zoz_T(t,zozs):
    zozs=np.array(zozs)
    if len(zozs.shape)==4:
        zozs=np.array([zozs]*t)
    zozs=[z.transpose([2,3,0,1]) for z in zozs]
    zozs[-1]=np.tensordot(zozs[-1],np.eye(2).ravel(),axis=((-1,),(0,)))
    ret=tt.fromproduct(zozs)
    return ret

def zoz_open_boundary_im(t):
    return tt.ones((4**t),cluster=(((4,),)*t))

def zoz_perfect_dephaser_im(t):
    return zoz_dephaser_im(t,1.0)
def zoz_dephaser_im(t,gamma):
    return tt.fromproduct([[1,1-gamma,1-gamma,1]]*t)
