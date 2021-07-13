import numpy as np
import tenpy.linalg.np_conserved as npc
from tenpy.linalg.charges import LegCharge
from tenpy.networks.site import Site
from tenpy.algorithms.exact_diag import ExactDiag
import tenpy
from functools import reduce
from abc import ABC,abstractmethod
class MPS(ABC):
    def __matmul__(self,other):
        if isinstance(other,MPS):
            return self.overlap(other)
        if isinstance(other,MPO):
            if isinstance(self,ProductMPS):
                return ProductMPS(other.transpose()@self.mpo,self.mps)
            return ProductMPS(other.transpose(),self)

    def overlap(self,other):
        if isinstance(other,ProductMPS) and isinstance(other.mpo,SimpleMPO) and isinstance(self,SimpleMPS):
            return _tp_overlap_sandwich(self.tpmps,other.mpo.tpmpo,other.mps.tpmps)
        if isinstance(self,ProductMPS) and isinstance(self.mpo,SimpleMPO) and isinstance(other,SimpleMPS):
            return _tp_overlap_sandwich(self.mps.tpmps,self.mpo.tpmpo,other.tpmps)
        else:
            return _tp_overlap_plain(self.to_tenpy(),other.to_tenpy())
    @classmethod
    def from_matrices(cls,Bs,Svs=None,norm=1.0):
        sites=[]
        bss=[]
        for b in Bs:
            b=np.array(b)
            bss.append(b.transpose([2,0,1])) #for some stupid reason
            sites.append(Site(LegCharge.from_trivial(b.shape[2])))
        return SimpleMPS(tenpy.networks.mps.MPS.from_Bflat(sites,bss,Svs,form=None))
    @classmethod
    def from_tenpy(cls,tpmps):
        return SimpleMPS(tpmps)
    @classmethod
    def from_product_state(cls,vs):
        return cls.from_matrices([np.array([[v]]) for v in vs])
class ProductMPS(MPS):
    def __init__(self,mpo,mps):
        self.mpo=mpo
        self.mps=mps
        self.L=self.mps.L
    def to_dense(self):
        return self.contract().to_dense()
    def to_tenpy(self):
        return self.contract().to_tenpy()
    def contract(self,**kwargs):
        if isinstance(self.mpo, ProductMPO):
            mpos=self.mpo.mpos
        else:
            mpos=[self.mpo]
        mps=self.mps
        for mpo in mpos:
            mps=mpo.apply(mps,**kwargs)
        return mps

def _tp_overlap_sandwich(left,mpo,right):
    mpo.IdL[0]=0
    mpo.IdR[-1]=0
    left=left.copy()
    _tp_canonical_form(left)
    _tp_canonical_form(right)
    for i in range(left.L):
        left.get_B(i).conj(True,True).conj(False,True)
    return tenpy.networks.mpo.MPOEnvironment(left,mpo,right).full_contraction(0)*left.norm*right.norm
def _tp_canonical_form(tpmps):
    #apparently tenpy can only do canonical form for L>2 :(
    if tpmps.L>2:
        tpmps.canonical_form(False)
    elif tpmps.L==2:
        B1=tpmps.get_B(0)
        B2=tpmps.get_B(1)
        M=np.einsum("abc,bed->ce",B1,B2)
        U,S,V=la.svd(M)
        tpmps.set_B(0,U,form="A")
        tpmps.set_B(1,V,form="A")
        tpmps.set_SR(0,S)
    else:
        #well if L=1 it is already canonical but we still need to teach this to tenpy
        tpmps.set_B(0,tpmps.get_B(0,None),"A")
        pass
def _tp_overlap_plain(left,right):
    left=left.copy()
    _tp_canonical_form(left)
    _tp_canonical_form(right)
    for i in range(left.L):
        left.get_B(i).conj(True,True).conj(False,True)
    return left.overlap(right)
def _process_options(kwargs):
    if "chi_max" in kwargs.keys():
        kwargs.get("trunc_params",{})["chi_max"]=kwargs["chi_max"]
        del kwargs["chi_max"]
    kwargs["verbose"]=False
    kwargs["compression_method"]="SVD"
class _B_helper():
    def __init__(self,tpmps):
        self.tpmps=tpmps
    def __getitem__(self,i):
        return self.tpmps.get_B(i,copy=False)
    def __setitem__(self,i):
        self.tpmps.set_B(i)

class _S_helper():
    def __init__(self,tpmps):
        self.tpmps=tpmps
    def __getitem__(self,i):
        return self.tpmps.get_SR(i)
    def __setitem__(self,i):
        self.tpmps.set_SR(i)
class SimpleMPS(MPS):
    def __init__(self,tpmps):
        self.tpmps=tpmps
        self.B=_B_helper(tpmps)
        self.S=_S_helper(tpmps)
        _tp_canonical_form(tpmps)
        self.L=tpmps.L
    def get_B(self,i):
        return self.tpmps.get_B(i,copy=True).to_ndarray().transpose([0,2,1])
    def get_S(self,i):
        return self.tpmps.get_SR(i)
    def canonicalize(self):
        self.canonicalize(False)
    def contract(self,**kwargs):
        return self
    def compress(self,**kwargs):
        _process_options(kwargs)
        self.tpmps.compress(options=kwargs)
    def to_dense(self):
        psi = self.tpmps.get_theta(0, self.tpmps.L)
        psi = npc.trace(psi,'vL', 'vR')
        psi = psi.to_ndarray()
        return psi.ravel()*self.tpmps.norm
    def copy(self):
        return MPS.from_tenpy(self.tpmps.copy())
    def to_tenpy(self):
        return self.tpmps
    def conj(self):
        ret=self.copy()
        for i in range(self.L):
            ret.tpmps.get_B(i).conj(True,True).conj(False,True)
        return ret

    def to_mpo(self,split):
        normp=state.norm**(1/state.L)
        Bs=[state.get_B(i,copy=True,form=None) for i in range(state.L)]
        for B in Bs:
            B.ireplace_labels(["p","vL","vR"],["(p.p*)","wL","wR"])
        Bs=[B*normp for B in Bs]
        Bs=[B.split_legs() for B in Bs]
        ret=MPO(nsites,Bs)
        return ret

class MPO(ABC):
    def __matmul__(self,other):
        '''
            Implements lazy multiplication of two MPO's or of a MPO and a MPS
        '''
        if isinstance(other,ProductMPS):
            return ProductMPS(self@other.mpo,other.mps)
        elif isinstance(other,MPS):
            return ProductMPS(self,other)
        elif isinstance(self,ProductMPO) and isinstance(other,ProductMPO):
            return ProductMPO(self.mpos+other.mpos)
        elif isinstance(other,ProductMPO):
            return ProductMPO([self]+other.mpos)
        elif isinstance(self,ProductMPO):
            return ProductMPO(self.mpos+[other])
        else:
            return ProductMPO([self,other])
    @classmethod
    def from_matrices(cls,Ws):
        sites=[]
        wss=[]
        for w in Ws:
            assert w.shape[2]==w.shape[3] # For now only square MPO's are possible
            sites.append(Site(LegCharge.from_trivial(w.shape[2])))
            leg_i=LegCharge.from_trivial(w.shape[0])
            leg_o=LegCharge.from_trivial(w.shape[1])
            leg_p=LegCharge.from_trivial(w.shape[2])
            Wn=npc.Array.from_ndarray(w,[leg_i,leg_o.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"])
            wss.append(Wn)
        return SimpleMPO(tenpy.networks.mpo.MPO(sites,wss))

    @classmethod
    def from_tenpy(cls,tpmpo):
        return SimpleMPO(tpmpo)
    @classmethod
    def from_product_operator(cls,ops):
        return cls.from_matrices([np.array([[o]]) for o in ops])
class _W_helper():
    def __init__(self,tpmpo):
        self.tpmpo=tpmpo
    def __getitem__(self,i):
        self.tpmpo.get_W(i,False)
    def __setitem__(self,i):
        self.tpmpo.set_W(i)

class SimpleMPO(MPO):
    def __init__(self,tpmpo):
        self.tpmpo=tpmpo
        self.W=_W_helper(tpmpo)
        self.L=tpmpo.L
    def contract(self):
        return self #already contracted
    def get_W(self,i):
        return self.tpmpo.get_W(i,True).to_ndarray()
    def input_dims(self):
        return [M.shape[3] for M in self.Ms]
    def output_dims(self):
        return [M.shape[2] for M in self.Ms]
    def bond_dims(self):
        return [M.shape[0] for M in self.Ms]+[self.Ms[-1].shape[1]]
    def to_dense(self):
        self.tpmpo.IdL[0]=0
        self.tpmpo.IdR[-1]=0 #How is this my job?
        ed=ExactDiag.from_H_mpo(self.tpmpo)
        ed.build_full_H_from_mpo()
        nda=ed.full_H.to_ndarray()
        if nda.shape[-1]==1:
            nda=nda[:,:,0]
        return nda
    def to_tenpy(self):
        return self.tpmpo
    def apply(self,mps,**kwargs):
        '''
            consumes mps, returns mpo applied to mps
        '''
        _process_options(kwargs)
        self.tpmpo.IdL[0]=0
        self.tpmpo.IdR[-1]=0 #How is this my job?
        tpmps=mps.tpmps
        mps.tpmps=None
        _tp_canonical_form(tpmps)
        self.tpmpo.apply(tpmps,kwargs)
        return MPS.from_tenpy(tpmps)
    def get_Ws(self):
        return [self.get_W(i) for i in range(self.tpmpo.L)]

class ProductMPO(MPO):
    def __init__(self,mpos):
        self.mpos=mpos
    def contract(self,**kwargs):
        mpolist=[x.tpmpo for x in self.mpos]
        def _multiply_W(w1,w2):
            pre=npc.tensordot(w1,w2,axes=[("p*",),("p",)])
            pre=pre.combine_legs([(0,3),(1,4)])
            pre.ireplace_labels(["(?0.?3)","(?1.?4)"],["wL","wR"])
            return pre
        Wps=[[m.get_W(i) for m in mpolist] for i in range(mpolist[0].L)]
        return SimpleMPO(tenpy.networks.mpo.MPO(mpolist[0].sites,[reduce(_multiply_W,Wp) for Wp in Wps]))
def outer(mpss):
    for mps in mpss:
        assert isinstance(mps,SimpleMPS)
    Bs=[]
    for mps in mpss:
        Bs.extend([mps.get_B(i) for i in range(mps.L)])
    return MPS.from_matrices(Bs)


def kron(mpos):
    for mpo in mpos:
        assert isinstance(mpo,SimpleMPO)
    Ws=[]
    for mpo in mpos:
        Ws.extend([mps.get_W(i) for i in range(mpo.L)])
    return MPO.from_matrices(Ws)
