import ttarray as tt

def temporal_gauge(im):
    im.clearcenter()
    im=im.asmatrices_unchecked()
    postval=np.array([1.0])
    for i in range(len(im)-1,-1,-1):
        preval=np.einsum("c,abc,b->a",postval,im[i],[0.5,0.0,0.0,0.5])
        im[i]=np.einsum("a,c,abc->abc",1/preval,postval,im[i])

def brickwork_boundary_dm_evolution(im,chs,init=np.eye(2)/2):
    dms=[dense.operator_to_state(init)]
    dim=init.shape[0]
    if isinstance(ch,np.ndarray):
        ch=[ch for _ in range(im.L)]
    for i in range(im.L):
        bimc=im_channel_dense(im,i)
        dmsd=dms[-1].reshape((bimc.shape[1]//4,dim**2))
        dmsd=np.einsum("ab,cb->ca",ch[i],dmsd)
        dms.append(dmsd.ravel())
        dmsd=dmsd.reshape((bimc.shape[1],dim**2//4))
        dmsd=np.einsum("ab,bc->ac",bimc,dmsd)
        dms.append(dmsd.ravel())
    return [dense.state_to_operator(np.sum(d.reshape((d.shape[0]//(dim**2),dim**2)),axis=0)) for d in dms]


def ising_boundary_evolution(im,chs,init=np.eye(2)/2):
    dms=[dense.operator_to_state(init)]
    dim=init.shape[0]
    if isinstance(ch,np.ndarray):
        ch=[ch for _ in range(im.L)]
    for i in range(im.L):
        bimc=im_channel_dense(im,i)
        dmsd=dms[-1].reshape((bimc.shape[1]//4,dim**2))
        dmsd=np.einsum("ab,cb->ca",ch[i],dmsd)
        dms.append(dmsd.ravel())
        dmsd=dmsd.reshape((bimc.shape[1],dim**2//4))
        dmsd=np.einsum("ab,bc->ac",bimc,dmsd)
        dms.append(dmsd.ravel())

    return [dense.state_to_operator(np.sum(d.reshape((d.shape[0]//(dim**2),dim**2)),axis=0)) for d in dms]

def zoz_boundary_evolution(im,chs,init=np.eye(2)/2):
    pass

def brickwork_embedded_evolution(iml,chs,imr,init=np.eye(2)/2):
    dms=[dense.operator_to_state(init)]
    dim=init.shape[0]
    if isinstance(ch,np.ndarray):
        ch=[ch for _ in range(left.L)]
    for i in range(left.L):
        limc=im_channel_dense(left,i)
        limc=limc.reshape((limc.shape[0]//4,4,limc.shape[1]//4,4))
        rimc=im_channel_dense(right,i)
        rimc=rimc.reshape((rimc.shape[0]//4,4,rimc.shape[1]//4,4))
        dmsd=dms[-1].reshape((limc.shape[2]*rimc.shape[2],dim**2))
        dmsd=np.einsum("ab,cb->ca",ch[i],dmsd)
        dms.append(dmsd.ravel())
        dmsd=dmsd.reshape((limc.shape[2],rimc.shape[2],4,dim**2//4))
        dmsd=np.einsum("abcd,cfdh->afbh",limc,dmsd)
        dmsd=dmsd.reshape((limc.shape[0],rimc.shape[2],dim**2//4,4))
        dmsd=np.einsum("abcd,ecgd->eagb",rimc,dmsd)
        dms.append(dmsd.ravel())
    return [dense.state_to_operator(np.sum(d.reshape((d.shape[0]//dim**2,dim**2)),axis=0)) for d in dms]

def ising_embedded_evolution(iml,chs,imr,init=np.eye(2)/2):
    pass

def zoz_embedded_evolution(iml,chs,imr,init=np.eye(2)/2):
    pass
