    @classmethod
    def load_from_hdf5(cls,hdf5obj,name=None):
        if name is not None:
            hdf5obj=hdf5obj[name]
        Ws=[]
        for i in range(int(np.array(hdf5obj["L"]))):
            Ws.append(np.array(hdf5obj["M_%i"%i]))
        return cls.from_matrices(Ws)


    def save_to_hdf5(self,hdf5obj,name=None):
        if name is not None:
            hdf5obj=hdf5obj.create_group(name)
        L=self.tpmps.L
        hdf5obj["L"]=L
        for i in range(L-1):
            hdf5obj["M_%i"%i]=np.array(self.get_B(i))
        hdf5obj["M_%i"%(L-1)]=np.array(self.get_B(L-1)*self.tpmps.norm)
    def load_from_hdf5(cls,hdf5obj,name=None):
        if name is not None:
            hdf5obj=hdf5obj[name]
        Ws=[]
        for i in range(int(np.array(hdf5obj["L"]))):
            Ws.append(np.array(hdf5obj["M_%i"%i]))
        return cls.from_matrices(Ws)
