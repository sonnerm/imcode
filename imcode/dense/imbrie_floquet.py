import scipy.sparse as sp
import pickle
import copy
import gmpy
import uuid
import numpy as np
import scipy.sparse.linalg as sla
import numpy.linalg as la
import scipy.linalg as scla
from .ed_utils import *
def calc_imbrie_floquet(L,h,g,j,H,G,J,T,inner):
    res=[]
    for i in range(inner):
        Hs=np.random.uniform(-H+h,H+h,L)
        Gs=np.random.uniform(-G+g,G+g,L)
        Js=np.random.uniform(-J+j,J+j,L)
        F=get_imbrie_F_p(Hs,Gs,Js,T)
        eigs,eigv=la.eigh(F+F.T.conj())
        eigs=np.log(np.diag(eigv.T.conj()@F@eigv)).imag
        sec=trivial_sector(L)
        fid=uuid.uuid4()
        ret={ "model":"imbrie_floquet",
        "H":{"random":"uniform","d":2*H,"m":h,"sample":Hs},
        "G":{"random":"uniform","d":2*G,"m":g,"sample":Gs},
        "J":{"random":"uniform","d":2*J,"m":j,"sample":Js},
        "drive":{"part":"G","T":T},
        "algorithm":"fd", "boundary":"P",
        "central_entropy":get_S(eigv.T,sec),"L":L,
        "floquet_energies":eigs,"uuid":fid}
        # hist_c,hist_sx,hist_sz=spectral_function(L,eigs,eigv)
        # ret["hist_sx"]=hist_sx
        # ret["hist_sz"]=hist_sz
        # ret["hist_c"]=hist_c
        ret["tau"],ret["ktau"]=ktau(eigs)
        res.append(ret)
        np.save(open("wfs/wfs_%s"%str(fid),"wb"),eigv.T)
        # save_ret=copy.deepcopy(ret)

        # save_ret["eigenstates"]=eigv.T
        # pickle.dump(save_ret,open("%s.pickle"%str(fid),"wb"))
    return res
