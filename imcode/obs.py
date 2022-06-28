import ttarray as tt

# def temporal_gauge(im):
    # im.clearcenter()
    # im=im.tomatrices_unchecked()
    # postval=np.array([1.0])
    # for i in range(len(im)-1,-1,-1):
    #     preval=np.einsum("c,abc,b->a",postval,im[i],[0.5,0.0,0.0,0.5])
    #     im[i]=np.einsum("a,c,abc->abc",1/preval,postval,im[i])


def zoz_tracevalues(im):
    return ising_tracevalues(im) # i think that is correct

def ising_tracevalues(im):
    im=im.tomatrices_unchecked()
    postval=[np.array([1.0])]
    for m in im[::-1]:
        postval.append(np.einsum("c,abc,b->a",postval[-1],m,[0.5,0.0,0.0,0.5]))
    return postval[::-1]

def brickwork_tracevalues(im):
    im=im.tomatrices_unchecked()
    postval=[np.array([1.0])]
    for m in im[::-1]:
        postval.append(np.einsum("d,abcd,b,c->a",postval[-1],m,[0.5,0,0,0.5],[1,0,0,1]))
    return postval[::-1]

def brickwork_boundary_dm_evolution(im,chs,init=np.eye(2)/2):
    dm=operator_to_state(init)[:,None]
    pvals=brickwork_tracevalues(im)
    yield dm[:,0]
    dm/=pvals[0]
    imm=im.tomatrices_unchecked()
    if isinstance(chs,np.ndarray) and len(chs.shape)==2:
        chs=[chs for _ in range(im.L)]
    for ch,m,pv in zip(chs,imm,pvals[1:]):
        dm=np.einsum("ab,bacd->cd",dm,m)
        yield np.einsum("ab,b->a",dm,pv)
        dm=np.einsum("ab,ca->cb",dm,ch)
        yield np.einsum("ab,b->a",dm,pv)


def ising_boundary_evolution(im,chs,init=np.eye(2)/2):
    dm=operator_to_state(init)[:,None]
    pvals=ising_tracevalues(im)
    yield dm[:,0]
    dm/=pvals[0]
    imm=im.tomatrices_unchecked()
    if isinstance(chs,np.ndarray) and len(chs.shape)==2:
        chs=[chs for _ in range(im.L)]
    for ch,m,pv in zip(chs,imm,pvals[1:]):
        dm=np.einsum("ab,bad->ad",dm,m)
        yield np.einsum("ab,b->a",dm,pv)
        dm=np.einsum("ab,ca->cb",dm,ch)
        yield np.einsum("ab,b->a",dm,pv)

def zoz_boundary_evolution(im,ozs,init=np.eye(2)/2):
    dm=operator_to_state(init)[:,None]
    pvals=ising_tracevalues(im)
    yield dm[:,0]
    dm/=pvals[0]
    imm=im.tomatrices_unchecked()
    if isinstance(ozs,np.ndarray) and len(chs.shape)==2:
        chs=[ozs for _ in range(im.L)]
    for ch,m,pv in zip(chs,imm,pvals[1:]):
        dm=np.einsum("ab,bcd,eac->ed",dm,m,ch)
        yield np.einsum("ab,b->a",dm,pv)

def brickwork_embedded_evolution(iml,chs,imr,init=np.eye(2)/2):
    dm=operator_to_state(init)[:,None,None]
    pvalsl=brickwork_tracevalues(iml)
    pvalsr=brickwork_tracevalues(imr)
    yield dm[:,0,0]
    dm/=pvalsl[0]
    dm/=pvalsr[0]
    imml=iml.tomatrices_unchecked()
    immr=imr.tomatrices_unchecked()
    if isinstance(chs,np.ndarray) and len(ozs.shape)==2:
        chs=[ozs for _ in range(im.L)]
    for ch,ml,mr,pvl,pvr in zip(chs,imml,immr,pvalsl[1:],pvalsr[1:]):
        dm=np.einsum("abc,bade->dec",dm,ml)
        dm=np.einsum("abc,cade->dbe",dm,mr)
        yield np.einsum("abc,b,c->a",dm,pvl,pvr)
        dm=np.einsum("abc,da->dbc",dm,ch)
        yield np.einsum("abc,b,c->a",dm,pvl,pvr)

def ising_embedded_evolution(iml,chs,imr,init=np.eye(2)/2):
    dm=operator_to_state(init)[:,None,None]
    pvalsl=ising_tracevalues(iml)
    pvalsr=ising_tracevalues(imr)
    yield dm[:,0,0]
    dm/=pvalsl[0]
    dm/=pvalsr[0]
    imml=iml.tomatrices_unchecked()
    immr=imr.tomatrices_unchecked()
    if isinstance(chs,np.ndarray) and len(chs.shape)==2:
        chs=[chs for _ in range(im.L)]
    for ch,ml,mr,pvl,pvr in zip(chs,imml,immr,pvalsl[1:],pvalsr[1:]):
        dm=np.einsum("abc,bae->aec",dm,ml)
        dm=np.einsum("abc,cae->abe",dm,mr)
        yield np.einsum("abc,b,c->a",dm,pvl,pvr)
        dm=np.einsum("abc,da->dbc",dm,ch)
        yield np.einsum("abc,b,c->a",dm,pvl,pvr)

def zoz_embedded_evolution(iml,zozs,imr,init=np.eye(2)/2):
    dm=operator_to_state(init)[:,None,None]
    pvalsl=zoz_tracevalues(iml)
    pvalsr=zoz_tracevalues(imr)
    yield dm[:,0,0]
    dm/=pvalsl[0]
    dm/=pvalsr[0]
    imml=iml.tomatrices_unchecked()
    immr=imr.tomatrices_unchecked()
    if isinstance(zozs,np.ndarray) and len(zozs.shape)==4:
        zozs=[zozs for _ in range(im.L)]
    for ch,ml,mr,pvl,pvr in zip(zozs,imml,immr,pvalsl[1:],pvalsr[1:]):
        dm=np.einsum("abc,bae,cae,->aec",dm,ml,mr,ch)
        yield np.einsum("abc,b,c->a",dm,pvl,pvr)
