import numpy as np
from imbrie_floquet import *
import os
import h5py

def write_dict_to_hdfobj(hdfobj,pre,d):
    if isinstance(d,dict):
        for k,v in d.items():
            write_dict_to_hdfobj(hdfobj,"%s/%s"%(pre,k),v)
    elif not isinstance(d,dict):
        # print((type(d),pre))
        hdfobj[pre]=d
def diagonalize(doc):
    np.random.seed(doc["seed"])
    h=doc["H"]["m"]
    H=doc["H"]["d"]/2
    assert doc["H"]["random"]=="uniform"
    g=doc["G"]["m"]
    G=doc["G"]["d"]/2
    assert doc["G"]["random"]=="uniform"
    j=doc["J"]["m"]
    J=doc["J"]["d"]/2
    assert doc["J"]["random"]=="uniform"
    L=doc["L"]
    assert doc["boundary"]=="P"
    assert doc["drive"]["part"]=="G"
    T=doc["drive"]["T"]
    Hs=np.random.uniform(-H+h,H+h,L)
    Gs=np.random.uniform(-G+g,G+g,L)
    Js=np.random.uniform(-J+j,J+j,L)
    F=get_imbrie_F_p(Hs,Gs,Js,T)
    eigs,eigv=la.eigh(F+F.T.conj())
    eigs=np.log(np.diag(eigv.T.conj()@F@eigv)).imag
    sd={"H":copy.copy(doc["H"]),"G":copy.copy(doc["G"]),"J":copy.copy(doc["J"])}
    sd["H"]["sample"]=Hs
    sd["G"]["sample"]=Gs
    sd["J"]["sample"]=Js
    sd["eigenvectors"]=eigv.T
    sd["floquet_energies"]=eigs
    with h5py.File(os.path.join("out","%s.h5"%(str(doc["_id"]))),"w") as f:
        write_dict_to_hdfobj(f,"",sd)
