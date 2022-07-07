
import numpy as np
from scipy import linalg
import sys
from create_correlation_block import create_correlation_block
from entropy import entropy
from tests import evolve_rho_T, evolve_rho, evolve_rho_CT
import h5py
import matplotlib.pyplot as plt

#filename = '/Users/julianthoenniss/Documents/PhD/data/compmode=G_o=1_Jx=0.1_Jy=0.1_g=0.0mu=0.0_del_t=1.0_beta=0.0_L=41_init=2'
#filename = '/Users/julianthoenniss/Documents/PhD/data/Jx=0.1_Jy=0.1_g=0.0mu=0.0_del_t=1.0_L=200_FermiSea'
#filename = '/Users/julianthoenniss/Documents/PhD/data/compmode=C_o=1_Jx=0.05_Jy=0.05_g=0.0mu=0.0_del_t=1.0_beta=0.0_L=200_init=2'
filename = '/Users/julianthoenniss/Documents/PhD/data/Millis_mu=0_timestep=0.05_test3'

#filename = '/Users/julianthoenniss/Documents/PhD/data/compmode=C_o=1_Jx=0.1_Jy=0.1_g=0.0mu=0.0_del_t=1.0_beta=0.0_L=20_init=2'
#filename = '/Users/julianthoenniss/Documents/PhD/data/compmode=C_Jx=0.1_Jy=0.1_g=0.0mu=0.0_del_t=1.0_beta=0.0_L=200_init=3'
delta_t = 0.05
conv = 'M'

if conv == 'J':
    filename += '_my_conv' 
    print('using Js convention')
elif conv == 'M':
    filename += '_Michaels_conv' 
    print('using Ms convention')


with h5py.File(filename+'_propag_test' + ".hdf5", 'w') as f:
        dset_propag_IM = f.create_dataset('propag_IM', (50,),dtype=np.complex_)
        dset_propag_exact = f.create_dataset('propag_exact', (50,),dtype=np.complex_)

trace_vals = []
trace_vals_const = []
rho_eigvals_min = []
rho_eigvals_max = []

max_time = 49
for iter in range(1,max_time):

    """
    with h5py.File(filename + '.hdf5', 'r') as f:
        times_read = f['temp_entr']
        nbr_Floquet_layers  = int(times_read[iter,0])
        print('times: ', nbr_Floquet_layers , 'iter: ', iter)
    iter_read = iter
    """

    nbr_Floquet_layers = iter
    iter_read = nbr_Floquet_layers
    

    B = np.zeros((4*nbr_Floquet_layers,4*nbr_Floquet_layers), dtype=np.complex_)
    with h5py.File(filename + '.hdf5', 'r') as f:
        B = f['IM_exponent'][iter_read,:4*nbr_Floquet_layers,:4*nbr_Floquet_layers]


    B_const = np.zeros((4*max_time,4*max_time), dtype=np.complex_)
    with h5py.File(filename + '.hdf5', 'r') as f:
        B_const = f['IM_exponent'][iter_read,:4*max_time,:4*max_time]

    dim_B = B.shape[0]
    dim_B_const = B_const.shape[0]


    if conv == 'M':
        U = np.zeros(B.shape)#order in the way specified in pdf for him (forward,backward,forward,backward,...)
        for i in range (nbr_Floquet_layers):
            U[4*i, B.shape[0] //2 - (2*i) -1] = 1
            U[4*i + 1, B.shape[0] //2 + (2*i)] = 1
            U[4*i + 2, B.shape[0] //2 - (2*i) -2] = 1
            U[4*i + 3, B.shape[0] //2 + (2*i) + 1] = 1
        B = U.T @ B @ U
        
        U = np.zeros(B_const.shape)#order in the way specified in pdf for him (forward,backward,forward,backward,...)
        for i in range (nbr_Floquet_layers):
            U[4*i, B_const.shape[0] //2 - (2*i) -1] = 1
            U[4*i + 1, B_const.shape[0] //2 + (2*i)] = 1
            U[4*i + 2, B_const.shape[0] //2 - (2*i) -2] = 1
            U[4*i + 3, B_const.shape[0] //2 + (2*i) + 1] = 1
        B_const = U.T @ B_const @ U
        
        print('Rotated from M basis to Grassmann basis')

    else:
    #rotate into correct basis with input/output variables
        S = np.zeros(B.shape,dtype=np.complex_)
        for i in range (dim_B//4):#order plus and minus next to each other
            S [dim_B // 2 - (2 * i) - 2,4 * i] = 1
            S [dim_B // 2 - (2 * i) - 1,4 * i + 2] = 1
            S [dim_B // 2 + (2 * i) ,4 * i + 1] = 1
            S [dim_B // 2 + (2 * i) + 1,4 * i + 3] = 1

        #the following two transformation bring it into in/out- basis (not theta, zeta)
        rot = np.zeros(B.shape,dtype=np.complex_)
        for i in range(0,dim_B, 2):
            rot[i,i] = 1./np.sqrt(2)
            rot[i,i+1] = 1./np.sqrt(2)
            rot[i+1,i] = - 1./np.sqrt(2) * np.sign(dim_B//2 - i-1)
            rot[i+1,i+1] = 1./np.sqrt(2) * np.sign(dim_B//2 - i-1)
        B = rot.T @ S @ B @ S.T @ rot 
        print('Rotated from J basis to Grassmann basis')

  
    #at this point, B is in the basis of the Grassmann approach (unfolded)


    #check trace preservation

    #for a single final-time IMs
    #add integration measure
    #measure (this imposes the IT initial state)
    B_ad = B
    #rho_exponent_evolved = evolve_rho_T(0,(B_ad),nbr_Floquet_layers, iter)
    print(B_ad)
    rho_exponent_evolved = evolve_rho(0,(B_ad))
    rho_evolved = np.zeros((2,2),dtype=np.complex_)
    rho_evolved[0,0] = 1
    rho_evolved[1,1] = 2 * rho_exponent_evolved[0,1]
    rho_evolved *= 0.5
    trace_vals.append(np.trace(rho_evolved))
    rho_eigvals = linalg.eigvals(rho_evolved)
    rho_eigvals_max.append(np.max(rho_eigvals))
    rho_eigvals_min.append(np.min(rho_eigvals))
    print(rho_evolved, np.trace(rho_evolved))

    
    B_ad = B_const
    
    rho_exponent_evolved = evolve_rho_T(0,B_ad,max_time, iter)
    rho_evolved = np.zeros((2,2),dtype=np.complex_)
    rho_evolved[0,0] = 1
    rho_evolved[1,1] = 2 * rho_exponent_evolved[0,1]
    rho_evolved *= 0.5
    trace_vals_const.append(np.trace(rho_evolved))
    print('trace2',np.trace(rho_evolved))
    print('eigvals2',linalg.eigvals(rho_evolved))
    

    #add integration measure
    #measure (this imposes the IT initial state)
    for i in range (dim_B//2-1):
        B[2 * i + 1 ,2 * i+2 ] += 1
        B[2 * i + 2 ,2 * i+1 ] -= 1

    #temporal boundary condition for measure
    B[0, dim_B - 1] += 1#sign because substituted in such a way that all kernels are the same.
    B[dim_B - 1,0] -= 1

    

    B_inv = linalg.inv(B)

    with h5py.File(filename+'_propag_test' + ".hdf5", 'a') as f:
        propag_data = f['propag_IM']
        propag_data[iter] = - B_inv[0,dim_B//2-1]
   
    print(- B_inv[0,dim_B//2-1])
print(trace_vals)
plt.plot(np.arange(1,48)*delta_t, trace_vals[:47],linewidth=2,label='Tr'+r'$(\rho)$')
plt.plot(np.arange(1,48)*delta_t, rho_eigvals_max[:47],linewidth=2,label='max. eigenvalue of '+ r'$\rho$')
plt.plot(np.arange(1,48)*delta_t, rho_eigvals_min[:47],linewidth=2,label='min. eigenvalue of '+ r'$\rho$')
#plt.plot(np.arange(2,28)*delta_t, trace_vals_const[1:27],linewidth=2,linestyle = '--',label='fixed IM at ' + r'$T=29$')
plt.plot(np.arange(1,48)*delta_t, 47*[1.0], linestyle='--',color="grey")
plt.xlabel('physical time '+r'$t$')
#plt.ylabel('Tr'+r'$(\rho)$')
plt.text(.1,0.6,r'$ \delta t = 0.01$'+ 'spinless, Inf. Temp., no local dynamics')
#plt.ylim([0.7,1.3])
plt.legend()
plt.show()