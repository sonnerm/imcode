from .. import MPO
def im_channel(im,t):
    imW=im.get_B(t)[None,:,:,:].transpose([0,3,1,2])
    idW=np.einsum("ac,cd->acd",np.eye(4),np.eye(4))[:,None,:,:]
    return MPO.from_matrices([imW,idW])
