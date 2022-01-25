import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import scipy.optimize
import h5py

def alg_decay(x,a,b):
    return a/x+b


sites = [100,200,500,1000]
filename = "FS_corr_delta=0.3"
t=1
delta = 0.3

for nsites in sites:

    H = np.zeros((nsites,nsites))

    for i in range (nsites-1):
        H[i,i+1] = -t * (1+delta/2 * (-1)**i)
    H += H.T.conj()


    eigvals, U = linalg.eigh(H)

    diag_H =U.T.conj() @ H @ U

    half_filling_identity = np.identity(nsites)
    for i in range (nsites//2):
        half_filling_identity[nsites//2+i,nsites//2+i] = 0
    print(half_filling_identity)
    Lambda = U.conj() @ half_filling_identity @ U.T
    #print(np.arange(nsites),Lambda[0,:])


    
    with h5py.File(filename + ".hdf5", 'w') as f:
        dset_corr = f.create_dataset('FS_corr_L='+ str(nsites), (nsites,nsites),dtype=np.float_)
        dset_corr[:,:] = Lambda
    
    """
    filename = "FS_corr_delta=0.3"
    Lambda = np.zeros((nsites,nsites))
    with h5py.File(filename + '.hdf5', 'r') as f:
        corr_data = f['FS_corr_L='+ str(nsites)]
        Lambda = corr_data[:,:]
    """


    x = np.arange(nsites)
    fig, ax = plt.subplots()
    ax.plot(x,abs(Lambda[0,:]))

    p0 = (1,0)
    params, cv = scipy.optimize.curve_fit(alg_decay, x[np.arange(5,nsites,2)], abs(Lambda[0,range(5,nsites,2)]), p0)
    a ,b= params
    print(a,b)
        
    ax.plot(x, alg_decay(x,a,b),'--',label= 'alg. fit: '+ r'${}/ x+{}$'.format(round(a,6),round(b,6)))
    plt.show()