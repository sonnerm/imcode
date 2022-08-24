import imcode
import pytest
import ttarray as tt
import numpy as np
from imcode import SZ,SX,ID,ZE
from .. import check_model
def test_simple(seed_rng):
    Te=imcode.heisenberg_Te(1,0,0,0)
    To=imcode.heisenberg_To(1,0,0,0)
    print(Te,To)
    print(list(imcode.brickwork_lcga([Te,To])))
# @pytest.mark.skip
def test_product_homhom(seed_rng):
    L=7
    t=5
    Jx,Jy,Jz,hx,hy,hz=np.random.random((6,))-0.5
    hx,hy,hz=0,0,0
    # Jx,Jy,Jz=0,0,0
    init=[np.random.random((2,))+np.random.random((2,))*1.0j-0.5-0.5j for _ in range(L)]
    init=[i.T.conj()+i for i in init]
    init=[i/np.sqrt(np.sum(i.conj()*i)) for i in init]
    init=tt.fromproduct([np.outer(i.T.conj(),i) for i in init])
    F=imcode.heisenberg_F(L,Jx,Jy,Jz,hx,hy,hz)
    Fs=[F for _ in range(t)]
    Te=imcode.heisenberg_Te(t,Jx,Jy,Jz,hx,hy,hz)
    To=imcode.heisenberg_To(t,Jx,Jy,Jz,hx,hy,hz)
    Ts=[To if ti%2==0 else Te for ti in range(2*t)]
    ch1=np.array(imcode.unitary_channel(imcode.heisenberg_F(1,Jx,Jy,Jz,hx,hy,hz)))
    ch2=np.array(imcode.unitary_channel(imcode.heisenberg_F(2,Jx,Jy,Jz,hx,hy,hz)))
    #rectangle
    check_model(L,t,init,Fs,Ts,Ts,imcode.brickwork_lcga,ch1,ch2,ch1,ch2,imcode.brickwork_boundary_evolution,imcode.brickwork_embedded_evolution,np.eye(4).ravel())
    #lcga
    # Ts=[imcode.heisenberg_Te(t,Jx,Jy,Jz,hx,hy,hz) if t%2==0 else imcode.heisenberg_To(t,Jx,Jy,Jz,hx,hy,hz) for t in range(1,t+1)]+[T]*(L-t)
    # check_model(L,t,init,Fs,Ts,Ts,imcode.brickwork_lcga,ch1,ch2,ch1,ch2,imcode.brickwork_boundary_evolution,imcode.brickwork_embedded_evolution)
