import ttarray as tt
import math
import numpy as np
from .channel import vectorize_operator,unvectorize_operator
from .norm import brickwork_norm

def _get_mat(even,odd):
    ret=np.zeros((2,2,2))
    ret[1,0,1]=odd[0]
    ret[1,1,0]=odd[1]
    ret[0,0,0]=even[0]
    ret[0,1,1]=even[1]
    return ret
_FERMI_A=_get_mat([1,1],[1,1])
_FERMI_B=_get_mat([1,-1],[1,1])

def apply_im(im,state,ind_im,ind_ap,fermi_inds=None,fermi_op=None,count=1):
    pass
def zoz_tracevalues(im):
    return ising_tracevalues(im) # i think that is correct

def ising_tracevalues(im):
    im=im.tomatrices_unchecked()[1:]
    postval=[np.array([1.0])]
    for m in im[::-1]:
        postval.append(np.einsum("c,abc,b->a",postval[-1],m,[0.5,0.0,0.0,0.5],optimize=True))
    return postval[::-1]

def brickwork_tracevalues(im,dim=2):
    im=im.tomatrices_unchecked()[1:]
    itens=vectorize_operator(np.eye(dim)/dim)
    trens=vectorize_operator(np.eye(dim))
    postval=[np.array([1.0])]
    for m in im[::-1]:
        postval.append(np.einsum("d,abcd,b,c->a",postval[-1],m.reshape((m.shape[0],dim**2,dim**2,m.shape[-1])),itens,trens,optimize=True))
    return postval[::-1]

def brickwork_boundary_evolution(im,chs,init=np.eye(2)/2,fermionic=False,normalize=False):
    '''
        Picture: IM - Local evolution
    '''
    init=np.array(init)
    if len(init.shape)==2:
        init=init[None,...,None]
    if len(im.shape)!=1:
        raise ValueError("Influence matrix must be 1D but has has shape %s!"%im.shape)
    if im.shape[0]%init.shape[0]:
        raise ValueError("Influence matrix length %i must be compatible with left index of init %i!"%(im.shape[0],init.shape[0]))
    t=int(math.log2(im.shape[0]//init.shape[0])/4)
    im.recluster(((init.shape[0],),)+((16,),)*t)
    if fermionic:
        Omps=tt.frommatrices([_FERMI_A[0,...][None,...]]+[_FERMI_B,_FERMI_A]*(t*2-1)+[_FERMI_B[...,0][...,None]])
        im=im*Omps
    dm=np.tensordot(im.M[0],vectorize_operator(init),axes=((1,),(0,)))[0,:,:,0]
    if normalize:
        dm/=brickwork_norm(im)
    pvals=brickwork_tracevalues(im)
    yield unvectorize_operator(np.einsum("ba,b->a",dm,pvals[0],optimize=True))
    imm=im.tomatrices_unchecked()[1:]
    if isinstance(chs,np.ndarray) and len(chs.shape)==2:
        chs=[chs for _ in range(im.L)]
    for ch,m,pva,pvb in zip(chs,imm,pvals[:-1],pvals[1:]):
        dm=np.einsum("ba,ca->bc",dm,ch,optimize=True)
        yield np.array(unvectorize_operator(np.einsum("ba,b->a",dm,pva,optimize=True)))
        dm=dm.reshape((dm.shape[0],4,dm.shape[1]//4))
        dm=np.einsum("bax,bacd->dcx",dm,m.reshape((m.shape[0],4,4,m.shape[-1])),optimize=True)
        dm=dm.reshape((dm.shape[0],dm.shape[2]*4))
        yield np.array(unvectorize_operator(np.einsum("ba,b->a",dm,pvb,optimize=True)))


def ising_boundary_evolution(im,chs,init=np.eye(2)/2):
    '''
        Picture: IM - Local evolution
    '''
    init=np.array(init)
    if len(init.shape)==2:
        init=init[None,...,None]
    if len(im.shape)!=1:
        raise ValueError("Influence matrix must be 1D but has has shape %s!"%im.shape)
    if im.shape[0]%init.shape[0]:
        raise ValueError("Influence matrix length %i must be compatible with left index of init %i!"%(im.shape[0],init.shape[0]))
    im.recluster(((init.shape[0],),)+((4,),)*int(math.log2(im.shape[0]//init.shape[0])/2))
    dm=np.tensordot(im.M[0],vectorize_operator(init),axes=((1,),(0,)))[0,:,:,0]
    pvals=ising_tracevalues(im)
    yield unvectorize_operator(np.einsum("ba,b->a",dm,pvals[0],optimize=True))
    imm=im.tomatrices_unchecked()[1:]
    if isinstance(chs,np.ndarray) and len(chs.shape)==2:
        chs=[chs for _ in range(im.L)]
    for ch,m,pva,pvb in zip(chs,imm,pvals[:-1],pvals[1:]):
        dm=np.einsum("ba,ca->bc",dm,ch,optimize=True)
        yield np.array(unvectorize_operator(np.einsum("ba,b->a",dm,pva,optimize=True)))
        dm=dm.reshape((dm.shape[0],4,dm.shape[1]//4))
        dm=np.einsum("bax,bad->dax",dm,m,optimize=True)
        dm=dm.reshape((dm.shape[0],dm.shape[2]*4))
        yield np.array(unvectorize_operator(np.einsum("ba,b->a",dm,pvb,optimize=True)))

def zoz_boundary_evolution(im,ozs,init=np.eye(2)/2):
    dm=np.array(vectorize_operator(init))[:,None]
    pvals=ising_tracevalues(im)
    yield np.array(unvectorize_operator(dm[:,0]))
    dm=dm/pvals[0]
    imm=im.tomatrices_unchecked()
    if isinstance(ozs,np.ndarray) and len(chs.shape)==2:
        chs=[ozs for _ in range(im.L)]
    for ch,m,pv in zip(chs,imm,pvals[1:]):
        dm=np.einsum("ab,bcd,eac->ed",dm,m,ch,optimize=True)
        yield np.array(unvectorize_operator(np.einsum("ab,b->a",dm,pv,optimize=True)))

def brickwork_embedded_evolution(iml,chs,imr,init=np.eye(2)/2,fermionic=False,normalize=False):
    '''
        Picture IML - local evolution - IMR
    '''
    init=np.array(init)
    if len(init.shape)==2:
        init=init[None,...,None]
    if len(iml.shape)!=1:
        raise ValueError("Left Influence matrix must be 1D but has has shape %s!"%iml.shape)
    if len(imr.shape)!=1:
        raise ValueError("Right Influence matrix must be 1D but has has shape %s!"%imr.shape)
    if iml.shape[0]%(init.shape[0]):
        raise ValueError("Left Influence matrix length %i must be compatible with left index of init %i!"%(iml.shape[0],init.shape[0]))
    if imr.shape[0]%(init.shape[3]):
        raise ValueError("Right Influence matrix length %i must be compatible with left index of init %i!"%(imr.shape[0],init.shape[3]))
    tl=int(math.log2(iml.shape[0]//init.shape[0])/4)
    tr=int(math.log2(imr.shape[0]//init.shape[-1])/4)
    iml.recluster(((init.shape[0],),)+((16,),)*tl)
    imr.recluster(((init.shape[-1],),)+((16,),)*tr)
    if fermionic:
        Ompsl=tt.frommatrices([_FERMI_A[0,...][None,...]]+[_FERMI_B,_FERMI_A]*(tl*2-1)+[_FERMI_B[...,0][...,None]])
        iml=Ompsl*iml
        Ompsr=tt.frommatrices([_FERMI_A[0,...][None,...]]+[_FERMI_B,_FERMI_A]*(tr*2-1)+[_FERMI_B[...,0][...,None]])
        imr=Ompsr*imr
    dm=np.einsum("abc,def,bge->cgf",iml.M[0],imr.M[0],vectorize_operator(init),optimize=True)
    pvalsl=brickwork_tracevalues(iml)
    pvalsr=brickwork_tracevalues(imr)
    if normalize:
        dm/=brickwork_norm(iml)
        dm/=brickwork_norm(imr)
    yield unvectorize_operator(np.einsum("bac,b,c->a",dm,pvalsl[0],pvalsr[0],optimize=True))
    imml=iml.tomatrices_unchecked()[1:]
    immr=imr.tomatrices_unchecked()[1:]
    if isinstance(chs,np.ndarray) and len(chs.shape)==2:
        chs=[chs for _ in range(iml.L)]
    for ch,ml,mr,pvla,pvra,pvlb,pvrb in zip(chs,imml,immr,pvalsl[:-1],pvalsr[:-1],pvalsl[1:],pvalsr[1:]):
        dm=np.einsum("bac,da->bdc",dm,ch,optimize=True)
        yield unvectorize_operator(np.einsum("bac,b,c->a",dm,pvla,pvra,optimize=True))
        dm=dm.reshape((dm.shape[0],4,dm.shape[1]//4,dm.shape[2]))
        dm=np.einsum("baxc,bane->enxc",dm,ml.reshape((ml.shape[0],4,4,ml.shape[-1])),optimize=True)
        dm=dm.reshape((dm.shape[0],dm.shape[2],4,dm.shape[3]))
        dm=np.einsum("bxac,cane->bxne",dm,mr.reshape((mr.shape[0],4,4,mr.shape[-1])),optimize=True)
        dm=dm.reshape((dm.shape[0],dm.shape[1]*4,dm.shape[3]))
        yield unvectorize_operator(np.einsum("bac,b,c->a",dm,pvlb,pvrb,optimize=True))

def brickwork_multi_evolution(init,ims):
    init=np.array(init)
    if len(init.shape)!=2:
        raise NotImplementedError("entangled initial states not yet implemented for multi-terminal setups")
    Li=int(math.log2(init.shape[0]))
    pvals=[]
    dm=vectorize_operator(init)
    tmin=0
    for i,imind in enumerate(ims):
        if len(imind[0].shape)!=1:
            raise ValueError("Influence matrix %i must be 1D but has has shape %s!"%(i,imind[0].shape))
        if len(imind)==3:
            ldim=imind[2]
        else:
            ldim=2
        t=int(round(math.log2(imind[0].shape[0])/math.log2(ldim**4)))
        if tmin:
            tmin=min(tmin,t)
        else:
            tmin=t
        imind[0].recluster(((1,),)+((ldim**4,),)*t)
        pvals.append(brickwork_tracevalues(imind[0],dim=ldim))
    dm=np.expand_dims(dm,tuple(range(1,len(ims)+1)))
    cpvals=[p[0] for p in pvals]
    def _calc_imp_dm(dm,pvals):
        dmr=np.copy(dm)
        for pv in pvals:
            dmr=np.tensordot(dmr,pv,((1,),(0,)))
        return unvectorize_operator(dmr)
    yield _calc_imp_dm(dm,cpvals)
    for ct in range(tmin):
        for i,imind in enumerate(ims):
            im=imind[0]
            pdim=imind[1]
            if len(imind)==3:
                ldim=imind[2]
            else:
                ldim=2
            dm=dm.reshape((pdim**2,ldim**2,4**(Li)//(pdim**2*ldim**2))+dm.shape[1:])
            imm=im.M[ct+1]
            imm=imm.reshape((imm.shape[0],ldim**2,ldim**2,imm.shape[-1]))
            dm=np.tensordot(dm,imm,((i+3,1),(0,1)))
            dm=dm.transpose((0,len(dm.shape)-2,1)+tuple(range(2,i+2))+(len(dm.shape)-1,)+tuple(range(i+2,len(ims)+1)))
            dm=dm.reshape((4**Li,)+dm.shape[3:])
            cpvals[i]=pvals[i][ct+1]
            yield _calc_imp_dm(dm,cpvals)

def ising_embedded_evolution(iml,chs,imr,init=np.eye(2)/2):
    '''
        Picture IML - local evolution - IMR
    '''
    init=np.array(init)
    if len(init.shape)==2:
        init=init[None,...,None]
    if len(iml.shape)!=1:
        raise ValueError("Left Influence matrix must be 1D but has has shape %s!"%iml.shape)
    if len(imr.shape)!=1:
        raise ValueError("Right Influence matrix must be 1D but has has shape %s!"%imr.shape)
    if iml.shape[0]%(init.shape[0]):
        raise ValueError("Left Influence matrix length %i must be compatible with left index of init %i!"%(iml.shape[0],init.shape[0]))
    if imr.shape[0]%(init.shape[3]):
        raise ValueError("Right Influence matrix length %i must be compatible with left index of init %i!"%(imr.shape[0],init.shape[3]))
    iml.recluster(((init.shape[0],),)+((4,),)*int(math.log2(iml.shape[0]//init.shape[0])/2))
    imr.recluster(((init.shape[-1],),)+((4,),)*int(math.log2(imr.shape[0]//init.shape[-1])/2))
    dm=np.einsum("abc,def,bge->cgf",iml.M[0],imr.M[0],vectorize_operator(init),optimize=True)
    pvalsl=ising_tracevalues(iml)
    pvalsr=ising_tracevalues(imr)
    yield unvectorize_operator(np.einsum("bac,b,c->a",dm,pvalsl[0],pvalsr[0],optimize=True))
    imml=iml.tomatrices_unchecked()[1:]
    immr=imr.tomatrices_unchecked()[1:]
    if isinstance(chs,np.ndarray) and len(chs.shape)==2:
        chs=[chs for _ in range(iml.L)]
    for ch,ml,mr,pvla,pvra,pvlb,pvrb in zip(chs,imml,immr,pvalsl[:-1],pvalsr[:-1],pvalsl[1:],pvalsr[1:]):
        dm=np.einsum("bac,da->bdc",dm,ch,optimize=True)
        yield unvectorize_operator(np.einsum("bac,b,c->a",dm,pvla,pvra))
        dm=dm.reshape((dm.shape[0],4,dm.shape[1]//4,dm.shape[2]))
        dm=np.einsum("baxc,bae->eaxc",dm,ml,optimize=True)
        dm=dm.reshape((dm.shape[0],dm.shape[2],4,dm.shape[3]))
        dm=np.einsum("bxac,cae->bxae",dm,mr,optimize=True)
        dm=dm.reshape((dm.shape[0],dm.shape[1]*4,dm.shape[3]))
        yield unvectorize_operator(np.einsum("bac,b,c->a",dm,pvlb,pvrb,optimize=True))

def zoz_embedded_evolution(iml,zozs,imr,init=np.eye(2)/2):
    dm=np.array(vectorize_operator(init))[:,None,None]
    pvalsl=zoz_tracevalues(iml)
    pvalsr=zoz_tracevalues(imr)
    yield np.array(unvectorize_operator(dm[:,0,0]))
    dm=dm/pvalsl[0]
    dm=dm/pvalsr[0]
    imml=iml.tomatrices_unchecked()
    immr=imr.tomatrices_unchecked()
    if isinstance(zozs,np.ndarray) and len(zozs.shape)==4:
        zozs=[zozs for _ in range(iml.L)]
    for ch,ml,mr,pvl,pvr in zip(zozs,imml,immr,pvalsl[1:],pvalsr[1:]):
        dm=np.einsum("abc,bae,cae,->aec",dm,ml,mr,ch,optimize=True)
        yield np.array(unvectorize_operator(np.einsum("abc,b,c->a",dm,pvl,pvr,optimize=True)))
