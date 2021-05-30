class MPS:
    def __matmul__(self,other):
        if isinstance(other,MPS):
            self.overlap(self,other)
        if isinstance(other,MPO):
            if isinstance(self,ProductMPS):
                return ProductMPS(other@self.mpo,self.mps)
            return ProductMPS(other,self)
class ProductMPS(MPS):
    def __init__(self,mpo,mps):
        pass
    def to_dense(self):
        return self.contract().to_dense()
    def to_tenpy(self):
        return self.contract().to_tenpy()

class SimpleMPS(MPS):
    def __init__(self,Bs,Ss=None,norm=1.0,canonicalize=0):
        pass
    def canonicalize(self):
        pass
    def from_product_state(self,vs):
        pass
    def contract(self):
        return self
    def compress(self,dim):
        pass
    def to_dense(self):
        pass
    def to_tenpy(self):
        pass
    def from_tenpy(tpmps):
        pass
    def to_mpo(self,split):
        pass
class TenpyMPS(MPS):
    def __init__(self,tp):
        self.tpmps=tpmps
    def canonicalize(self):
        self.tpmps.canonicalize(False)
    def contract(self):
        return self
    def compress(self):
        pass
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
