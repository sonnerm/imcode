from tenpy
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
            bss.append(npc.)
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
            return MPOEnvironment(stpmps,otpmpo,otpmps).full_contraction(0)*stpmps.norm*otpmps.norm
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
