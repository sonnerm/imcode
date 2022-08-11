import pytest
import imcode
import math
import ttarray as tt
import numpy as np
def calc_rdm(inds,dm):
    return np.array(tt.frommatrices([x if i in inds else np.trace(x,axis1=1,axis2=2).reshape((x.shape[0],1,1,x.shape[-1])) for i,x in enumerate(dm.M)]))
def check_model(L,t,init,Fs,Tsl,Tsr,lcga_fun,chl,chr2,che,che2,boundary_obs_fun,embedded_obs_fun,obc):
    #tebd dm
    mzzl,mzzr2,mzze,mzze2=[],[],[],[]
    minit=init
    for F in Fs:
        mzzl.append(calc_rdm([0],minit))
        mzzr2.append(calc_rdm([L-2,L-1],minit))
        mzze.append(calc_rdm([L//2],minit))
        mzze2.append(calc_rdm([L//2,L//2+1],minit))
        minit=(F@minit@F.T.conj())
        minit.truncate(chi_max=64)
    iml=list(lcga_fun(Tsl,init,boundary=tt.fromproduct([obc]*int(math.log2(Tsl[0].shape[1])/math.log2(obc.shape[0]))),chi_max=64))
    print([imcode.brickwork_norm(i) for i in iml])
    revinit=tt.frommatrices([i.transpose([3,1,2,0]) for i in init.tomatrices()[::-1]])
    imr=list(lcga_fun(Tsr,revinit,boundary=tt.fromproduct([obc]*int(math.log2(Tsl[0].shape[1])/math.log2(obc.shape[0]))),chi_max=64))
    if init.M[0].shape[0]==1 and init.M[0].shape[-1]==1:
        izzl=list(boundary_obs_fun(imr[L-2],chl,init=init.M[0][0,...,0]))
    else:
        izzl=list(boundary_obs_fun(imr[L-2],chl,init=init.M[0].transpose([3,1,2,0])))
    r2init=tt.frommatrices_slice([init.M[-2],init.M[-1]]).todense()
    izzr2=list(boundary_obs_fun(iml[L-3],chr2,init=r2init))
    if init.M[L//2].shape[0]==1 and init.M[L//2].shape[-1]==1:
        izze=list(embedded_obs_fun(iml[L//2-1],che,imr[L-L//2-2],init=init.M[L//2][0,...,0]))
    else:
        izze=list(embedded_obs_fun(iml[L//2-1],che,imr[L-L//2-2],init=init.M[L//2]))
    izze2=list(embedded_obs_fun(iml[L//2-1],che2,imr[L//2-2],init=tt.frommatrices_slice([init.M[L//2],init.M[L//2+1]])))
    for i,m in zip(izzl[::2],mzzl):
        assert i==pytest.approx(m)
    for i,m in zip(izzr2[::2],mzzr2):
        assert i==pytest.approx(m)
    for i,m in zip(izze[::2],mzze):
        assert i==pytest.approx(m)
    for i,m in zip(izze2[::2],mzze2):
        assert i==pytest.approx(m)
