import ttarray as tt
import numpy as np
def savehdf5(obj,hdf5obj,name,legacy=False):
    if legacy:
        if isinstance(obj,TensorTrainArray) and len(obj.shape)==1:
            L=obj.L
            hdf5obj["L"]=L
            Ms=obj.tomatrices_unchecked()
            for i,m in enumerate(Ms):
                hdf5obj["M_%i"%i]=np.array(m).transpose([0,2,1])
        elif isinstance(obj,TensorTrainArray) and len(obj.shape)==2:
            L=obj.L
            hdf5obj["L"]=L
            Ms=obj.tomatrices_unchecked()
            for i,m in enumerate(Ms):
                hdf5obj["M_%i"%i]=np.array(m).transpose([0,3,2,1])
        else:
            raise ValueError("legacy mode only allowed for TensorTrainArray with 1 or 2 dimensions")
    else:
        tt.savehdf5(obj,hdf5obj,name)


def loadhdf5(hdf5obj,name=None):
    try:
        return tt.loadhdf5(hdf5obj,name)
    except ValueError as e:
        if "type" in hdf5obj.keys():#not legacy mode,reraise
            raise e
        else:#legacy mode
            if name is not None:
                hdf5obj=hdf5obj[name]
            L=np.array(hdf5obj["L"])
            Ms=[]
            for i in range(L):
                m=np.array(hdf5obj["M_%i"%i])
                if len(m.shape)==3:
                    Ms.append(m.transpose([0,2,1]))
                elif len(m.shape)==4:
                    Ms.append(m.transpose([0,3,2,1]))
                else:
                    raise ValueError("Incompatible shape for legacy mode")
            return tt.frommatrices(Ms)
