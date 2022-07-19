import imcode
import pytest
import ttarray as tt
import numpy as np
from imcode import SZ,SX,ID,ZE
from .. import check_model
def test_product_homhom(seed_rng):
    L=7
    t=6
    Jx,Jy,Jz,hx,hy,hz=np.random.random((6,))-0.5
    init=[np.random.random((2,))+np.random.random((2,))*1.0j-0.5-0.5j for _ in range(L)]
    init=[i.T.conj()+i for i in init]
    init=[i/np.sqrt(np.sum(i.conj()*i)) for i in init]
    init=tt.fromproduct([np.outer(i.T.conj(),i) for i in init])
    F=imcode.heisenberg_F(L,Jx,Jy,Jz,hx,hy,hz)
    Fs=[F for _ in range(t)]
    T=imcode.heisenberg_T(t,Jx,Jy,Jz,hx,hy,hz)
    Ts=[T for _ in range(t)]
    ch1=np.array(imcode.unitary_channel(imcode.heisenberg_F(1,Jx,Jy,Jz,hx,hy,hz)))
    ch2=np.array(imcode.unitary_channel(imcode.heisenberg_F(2,Jx,Jy,Jz,hx,hy,hz)))
    #rectangle
    check_model(L,t,init,Fs,Ts,Ts,imcode.brickwork_lcga,ch1,ch2,ch1,ch2,imcode.brickwork_boundary_evolution,imcode.brickwork_embedded_evolution)
    #lcga
    Ts=[imcode.heisenberg_T(t,J,g,h) for t in range(1,t+1)]+[T]*(L-t)
    check_model(L,t,init,Fs,Ts,Ts,imcode.brickwork_lcga,ch1,ch2,ch1,ch2,imcode.brickwork_boundary_evolution,imcode.brickwork_embedded_evolution)
