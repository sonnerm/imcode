import tenpy.linalg.np_conserved as npc
import tenpy
class MPS:
    def __matmul__(self,other):
        if isinstance(other,MPS):
            self.overlap(self,other)
        if isinstance(other,MPO):
            if isinstance(self,ProductMPS):
                return ProductMPS(other@self.mpo,self.mps)
            return ProductMPS(other,self)
    def from_matrices(self,Bs,Svs):
        pass
    def from_tenpy(self,tpmps):
        return SimpleMPS()
    def from_product_state(self,vs):
        pass
class ProductMPS(MPS):
    def __init__(self,mpo,mps):
        self.mpo=mpo
        self.mps=mps
    def to_dense(self):
        return self.contract().to_dense()
    def to_tenpy(self):
        return self.contract().to_tenpy()
    def contract(self):
        if isinstance(self.mpo, ProductMPO):
            mpos=self.mpo.mpos
        else:
            mpos=[self.mpo]
        mps=self.mps
        for mpo in mpos:
            mps=mpo.apply(mps)
        return mps


class SimpleMPS(MPS):
    def __init__(self,Bs,Ss=None,norm=1.0,canonicalize=0):
        sites=[]
        bss=[]
        for b in Bs:
            sites.append(LegCharge.from_trivial(b.shape[2]))
            leg_i=LegCharge.from_trivial(b.shape[0])
            leg_o=LegCharge.from_trivial(b.shape[1])
            leg_p=LegCharge.from_trivial(b.shape[2])
            Bn=npc.Array.from_ndarray(b,[leg_i,leg_o.conj(),leg_p],labels=["vL","vR","p"])
            bss.append(Bn)
        return tenpy.networks.mps.MPS(sites,bss,Ss,norm,canonicalize)
    def canonicalize(self):
        self.canonicalize
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
            for i in range(im.L):
                stpmps.get_B(i).conj(True,True).conj(False,True)
            return stpmps.overlap(otpmps)
    def contract(self):
        return self
    def compress(self,dim):
        pass
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

class MPO:
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
    def from_matrices(Ws):
        return SimpleMPO(Ws)

    @classmethod
    def from_tenpy(tpmps):
        return SimpleMPO(tpmps)
    @classmethod
    def from_product_operator(ops):
        pass
class SimpleMPO(MPO):
    def __init__(self,Ws):
        sites=[]
        wss=[]
        for w in Ws:
            assert w.shape[2]==w.shape[3] # For now only square MPO's are possible
            sites.append(LegCharge.from_trivial(w.shape[2]))
            leg_i=LegCharge.from_trivial(w.shape[0])
            leg_o=LegCharge.from_trivial(w.shape[1])
            leg_p=LegCharge.from_trivial(w.shape[2])
            Wn=npc.Array.from_ndarray(w,[leg_i,leg_o.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"])
            wss.append(Wn)
        return tenpy.networks.mps.MPO(sites,wss)
    def contract(self):
        return self #already contracted
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
    def apply(self,mps,chi,options):
        '''
            consumes mps, returns mpo applied to mps
        '''
        self.tpmpo.IdL[0]=0
        self.tpmpo.IdR[-1]=0 #How is this my job?
        tpmps=mps.tpmps
        mps.tpmps=None
        if options is None:
            if chi is None:
                self.tpmpo.apply_naively(tpmps)
                tpmps.canonical_form(renormalize=False)
                return MPS.from_tenpy(tpmps)
            else:
                options={"trunc_params":{"chi_max":chi},"verbose":False,"compression_method":"SVD"}
        self.tpmpo.apply(tpmps,options)
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
    def concat(self,other):
        pass
    def outer(self,other):
        pass
class ProductMPO(MPO):
    def __init__(self,mpos):
        pass
    def contract(self):
        Wps=[[m.get_W(i) for m in self.mpos] for i in range(mpolist[0].L)]
        return MPO(mpolist[0].sites,[reduce(_multiply_W,Wp) for Wp in Wps])
    def _multiply_W(w1,w2):
        pre=npc.tensordot(w1,w2,axes=[("p*",),("p",)])
        pre=pre.combine_legs([(0,3),(1,4)])
        pre.ireplace_labels(["(?0.?3)","(?1.?4)"],["wL","wR"])
        return pre
    def concat(self,other):
        pass
    def outer(self,other):
        pass
