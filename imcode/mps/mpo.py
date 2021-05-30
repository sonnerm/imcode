class MPO:
    pass
class SimpleMPO(MPO):
    def __init__(self,Ms):
        self.Ms=Ms
    def contract(self):
        return self
    def to_mps(self,split=None):
        pass
    def input_dims(self):
        return [M.shape[3] for M in self.Ms]
    def output_dims(self):
        return [M.shape[2] for M in self.Ms]
    def bond_dims(self):
        return [M.shape[0] for M in self.Ms]+[self.Ms[-1].shape[1]]
    def check_consistency(self):
        pass
    def __matmul__(self,other):
        pass
    def __rmatmul__(self,other):
        pass
    def to_dense(self):
        pass
    def to_tenpy(self):
        pass
class ProductMPO(MPO):
    def __init__(self,mpos):
        pass
    def contract():
        pass
    def __matmul__(self,other):
        pass
    def __rmatmul__(self,other):
        pass
class TenpyMPO:
    def __init__(self,tpmpo):
        self.tpmpo=tpmpo
    def to_tenpy(self):
        return self.tpmpo
    def __matmul__(self,other):
        pass

    def __rmatmul__(self,other):
        pass
