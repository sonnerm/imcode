def boundary_dm_evolution(im,lop,init):
    pass
def embedded_dm_evolution(left,lop,right,init):
    pass
def boundary_z(im,lop,zs):
    """
        Multi-time z-diagonal correlation function
    """
    pass
def embedded_z(left,lop,right,zs):
    pass
def boundary_norm(im,lop):
    return boundary_z(im,lop,[(2,2)]*im.L)
def embedded_norm(left,lop,right):
    return boundary_z(left,lop,right,[(2,2)]*left.L)
