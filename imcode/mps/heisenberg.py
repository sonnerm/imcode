def heisenberg_F(Jx,Jy,Jz,hx,hy,hz):
    gates=[heisenberg_gate(jx,jy,jz) for (jx,jy,jz) in zip(Jx,Jy,Jz)]
    lop=[heisenberg_lop(chx,chy,chz) for (chx,chy,chz) in zip(hx,hy,hz)]
    return brickwork_F(gates,lop)
def heisenberg_gate(Jx,Jy,Jz):
    H=np.kron(SX,SX)*Jx+np.kron(SY,SY)*Jy+np.kron(SZ,SZ)*Jz
    return la.expm(1.0j*np.array(H))
def heisenberg_lop(hx,hy,hz):
    return la.expm(0.5j*(SX*hx+SY*hy+SZ*hz))
