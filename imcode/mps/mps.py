import numpy as np
import tenpy.linalg.np_conserved as npc
from tenpy.linalg.charges import LegCharge
from tenpy.networks.site import Site
from tenpy.algorithms.exact_diag import ExactDiag
import tenpy
from functools import reduce
from abc import ABC,abstractmethod
class MPS(ABC):
    # @abstractmethod
    # def __init__(self):
    #     pass
    def __matmul__(self,other):
        if isinstance(other,MPS):
            return self.overlap(other)
        if isinstance(other,MPO):
            if isinstance(self,ProductMPS):
                return ProductMPS(other@self.mpo,self.mps)
            return ProductMPS(other,self)
    # def outer(self,other):
    #     if isinstance(self,ProductMPS):

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
        self.tpmps.get_B(i,copy=False)
    def __setitem__(self,i):
        self.tpmps.set_B(i)

class _S_helper():
    def __init__(self,tpmps):
        self.tpmps=tpmps
    def __getitem__(self,i):
        self.tpmps.get_SR(i)
    def __setitem__(self,i):
        self.tpmps.set_SR(i)
class SimpleMPS(MPS):
    def __init__(self,tpmps):
        self.tpmps=tpmps
        self.B=_B_helper(tpmps)
        self.S=_S_helper(tpmps)
        self.tpmps.canonical_form(False)
    def get_B(self,i):
        self.tpmps.get_B(i,copy=True).to_ndarray()
    def get_S(self,i):
        self.tpmps.get_SR(i)
    def canonicalize(self):
        self.canonicalize(False)
    def overlap(self,other):
        if isinstance(other,ProductMPS):
            otpmpo=other.mpo.to_tenpy()
            otpmpo.IdL[0]=0
            otpmpo.IdR[-1]=0
            otpmps=other.mps.to_tenpy()
            stpmps=self.tpmps.copy()
            for i in range(stpmps.L):
                stpmps.get_B(i).conj(True,True).conj(False,True)
            return tenpy.networks.mpo.MPOEnvironment(stpmps,otpmpo,otpmps).full_contraction(0)*stpmps.norm*otpmps.norm
        else:
            otpmps=other.to_tenpy()
            stpmps=self.tpmps.copy()
            stpmps.canonical_form(False)
            for i in range(stpmps.L):
                stpmps.get_B(i).conj(True,True).conj(False,True)
            return stpmps.overlap(otpmps)
    def contract(self,**kwargs):
        return self
    def compress(self,**kwargs):
        _process_options(kwargs)
        self.tpmps.compress(options=kwargs)
    def to_dense(self):
        psi = self.tpmps.get_theta(0, self.tpmps.L)
        psi = npc.trace(self.psi,'vL', 'vR')
        psi = psi.to_ndarray()
        return psi.ravel()*self.tpmps.norm
    def to_tenpy(self):
        return self.tpmps

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
    def contract(self):
        return self #already contracted
    def get_W(self,i):
        return self.tpmpo.get_W(i,True).to_ndarray()
    def to_mps(self,split=None):
        nsites=[OperatorSite(s) for s in mpo.sites]
        Ws=[mpo.get_W(i,False) for i in range(mpo.L)]
        Ws=[W.combine_legs(["p","p*"],pipes=nsites[i].leg) for i,W in enumerate(Ws)]
        for W in Ws:
            W.ireplace_labels(["(p.p*)","wL","wR"],["p","vL","vR"])
        Svs=[np.ones(W.shape[0]) / np.sqrt(W.shape[1]) for W in Ws]
        Svs.append([1.0])
        Svs[0]=[1.0]
        ret=MPS(nsites,Ws,Svs,form=None)
        if mpo.L>2:
            ret.canonical_form(False)
        return ret
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
        tpmps.canonical_form(False)
        self.tpmpo.apply(tpmps,kwargs)
        return MPS.from_tenpy(tpmps)
    def to_mps(self):
        Ws=[self.tpmpo.get_W(i,False) for i in range(mpo.L)]
        Ws=[W.combine_legs(["p","p*"],pipes=nsites[i].leg) for i,W in enumerate(Ws)]
        for W in Ws:
            W.to_ndarray()
        ret=MPS(Ws)
        if mpo.L>2:
            ret.canonical_form(False)
        return ret
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
    def outer(self,other):
        pass
