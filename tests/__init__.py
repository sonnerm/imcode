import ttarray as tt
def calc_rdm(inds,dm):
    return np.array(tt.frommatrices([x if x in inds else np.trace(x,axis1=1,axis2=2) for x in dm.tomatrices()]))
def check_model(L,t,init,Fs,Tsl,Tsr,lcga_fun,chl,chr2,che,che2,boundary_obs_fun,embedded_obs_fun):
    #tebd dm
    mzzl,mzzr2,mzze,mzze2=[],[],[],[]
    minit=tt.fromproduct([np.outer(i.T.conj(),i) for i in init])
    for F in Fs:
        mzzl.append(calc_rdm([0],init))
        mzzr2.append(calc_rdm([L-2,L-1],minit))
        mzze.append(calc_rdm([L//2],minit))
        mzze2.append(calc_rdm([L//2,L//2+1],minit))
        minit=(F@minit@F.T.conj())
        minit.truncate(chi_max=64)
    iml=list(lcga_fun(Tsl,init,chi_max=64))
    imr=list(lcga_fun(Tsr,tt.fromproduct(init.tomatrices()[::-1]),chi_max=64))
    izzl=list(boundary_obs_fun(imr[L-2],chl,init=init.M[0]))
    izzr2=list(boundary_obs_fun(iml[L-3],chr2,init=np.array(tt.frommatrices_slice([init.M[-2],init.M[-1]]))))
    izze=list(embedded_obs_fun(imr[L-2],chl,init=init.M[0]))
    izze2=list(embedded_obs_fun(imr[L-2],chl,init=init.M[0]))
    for i,m in zip(izzl[::2],mzzl):
        assert i==pytest.approx(m)
    for i,m in zip(izzr2[::2],mzzr2): 
        assert i==pytest.approx(m)
    for i,m in zip(izze[::2],mzze):
        assert i==pytest.approx(m)
    for i,m in zip(izze2[::2],mzze2):
        assert i==pytest.approx(m)
