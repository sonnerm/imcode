
import numpy as np
from scipy import linalg
import sys
from create_correlation_block import create_correlation_block
from entropy import entropy
import h5py
import matplotlib.pyplot as plt
# seed the pseudorandom number generator
from random import seed
from random import random
# seed random number generator

np.set_printoptions(threshold=sys.maxsize, precision=1)
#filename = '/Users/julianthoenniss/Documents/PhD/data/compmode=G_o=1_Jx=0.1_Jy=0.1_g=0.0mu=0.0_del_t=1.0_beta=0.0_L=41_init=2'
filename = '/Users/julianthoenniss/Documents/PhD/data/interleaved_Jx=0.1_Jy=0.1_g=0.0mu=0.0_del_t=1.0_L=200InfTemp-FermiSea_my_conv'
filename = '/Users/julianthoenniss/Documents/PhD/data/Millis_mu=0_timestep=0.05_test3'
#filename = '/Users/julianthoenniss/Documents/PhD/data/compmode=C_o=1_Jx=0.1_Jy=0.1_g=0.0mu=0.0_del_t=1.0_beta=0.0_L=20_init=2'
#filename = '/Users/julianthoenniss/Documents/PhD/data/compmode=C_Jx=0.1_Jy=0.1_g=0.0mu=0.0_del_t=1.0_beta=0.0_L=200_init=3'

conv = 'M'

if conv == 'J':
    filename += '_my_conv' 
    print('using Js convention')
elif conv == 'M':
    filename += '_Michaels_conv' 
    print('using Ms convention')

t = 1. # hopping between spin species
delta_t = 0.05

with h5py.File(filename+'_spinfulpropag' + ".hdf5", 'w') as f:
    dset_propag_IM = f.create_dataset('propag_IM', (100,), dtype=np.complex_)
    dset_propag_exact = f.create_dataset(
        'propag_exact', (100,), dtype=np.complex_)

trace_vals = []
trace_vals_const = []
rho_eigvals_min = []
rho_eigvals_max = []
for iter in range(1,29):

    """
    with h5py.File(filename + '.hdf5', 'r') as f:
        times_read = f['temp_entr']
        nbr_Floquet_layers = int(times_read[iter, 0])
        print('times: ', nbr_Floquet_layers)
    """

    nbr_Floquet_layers = iter

    B = np.zeros((4*nbr_Floquet_layers, 4*nbr_Floquet_layers),
                 dtype=np.complex_)
    with h5py.File(filename + '.hdf5', 'r') as f:
        print(4*nbr_Floquet_layers, 4*nbr_Floquet_layers)
        B = f['IM_exponent'][iter, :4*nbr_Floquet_layers, :4*nbr_Floquet_layers]
   

    dim_B = B.shape[0]
 
    # rotate into Grassmann basis with input/output variables

    if conv == 'M':
        U = np.zeros(B.shape)#order in the way specified in pdf for him (forward,backward,forward,backward,...)
        for i in range (nbr_Floquet_layers):
            U[4*i, B.shape[0] //2 - (2*i) -1] = 1
            U[4*i + 1, B.shape[0] //2 + (2*i)] = 1
            U[4*i + 2, B.shape[0] //2 - (2*i) -2] = 1
            U[4*i + 3, B.shape[0] //2 + (2*i) + 1] = 1
        B = U.T @ B @ U
        
        
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


    #print(B)

    #here, the matrix B is in the Grassmann convention

    # adjust signs that make the influence matrix a vectorized state
    for i in range (dim_B):
        for j in range (dim_B):
            if (i+j)%2 == 1:
                B[i,j] *= -1



    exponent = np.zeros((4*dim_B, 4*dim_B), dtype=np.complex_)#this exponent will contain the exponents of both spin species as well as the impurity dynamics
    # Influence matices for both spin species
    exponent[dim_B:2*dim_B, dim_B:2*dim_B] = B[:, :]
    exponent[2*dim_B:3*dim_B, 2*dim_B:3*dim_B] = B[:, :]

    # integration measure
    # spin up
    exponent[:dim_B, 2*dim_B:3*dim_B] = np.identity(dim_B)
    exponent[2*dim_B:3*dim_B, :dim_B] = -np.identity(dim_B)
    # spin down
    exponent[dim_B:2*dim_B, 3*dim_B:] = np.identity(dim_B)
    exponent[3*dim_B:, dim_B:2*dim_B] = -np.identity(dim_B)


    # impurity
    #hopping between spin species (gate is easily found by taking kernel of xy gate at isotropic parameters):
    seed(10)
    for i in range(dim_B//4-1):
        t =  0.57*np.cos(0.42 * i)
        mu_up = 0.3*np.sin(2.2 * i)
        mu_down = 0.18*np.sin(1.82 * i)
        #mu_up =0# random()
        #mu_down =0# random()
        print(t,mu_up,mu_down)

        T=1+np.tan(t/2)**2
        # forward 
        # (matrix elements between up -> down)
        exponent[dim_B//2 - 3 - 2*i, 3*dim_B + dim_B//2 - 2 - 2*i] += 1.j * np.tan(t/2) *2/T *np.exp(1.j * mu_up)
        exponent[dim_B//2 - 2 - 2*i, 3*dim_B + dim_B//2 - 3 - 2*i] -= 1.j * np.tan(t/2)*2/T *np.exp(1.j * mu_down)
        #(matrix elements between up -> up)
        exponent[dim_B//2 - 3 - 2*i, dim_B//2 - 2 - 2*i] += 1 *np.cos(t) *np.exp(1.j * mu_up)
        exponent[3*dim_B + dim_B//2 - 3 - 2*i, 3*dim_B + dim_B//2 - 2 - 2*i] += 1 *np.cos(t) *np.exp(1.j * mu_down)

        # forward Transpose (antisymm)
        exponent[3*dim_B + dim_B//2 - 2 - 2*i, dim_B//2 - 3 - 2*i] += -1.j * np.tan(t/2)*2/T *np.exp(1.j * mu_up)
        exponent[3*dim_B + dim_B//2 - 3 - 2*i, dim_B//2 - 2 - 2*i] -= -1.j * np.tan(t/2)*2/T *np.exp(1.j * mu_down)
        exponent[dim_B//2 - 2 - 2*i,dim_B//2 - 3 - 2*i] += -1 *np.cos(t) *np.exp(1.j * mu_up)
        exponent[3*dim_B + dim_B//2 - 2 - 2*i, 3*dim_B + dim_B//2 - 3 - 2*i] += -1 *np.cos(t) *np.exp(1.j * mu_down)

        # backward
        exponent[dim_B//2 + 1 + 2*i, 3*dim_B + dim_B//2 + 2 + 2*i] += - 1.j * np.tan(t/2)*2/T *np.exp(-1.j * mu_down)
        exponent[dim_B//2 + 2 + 2*i, 3*dim_B + dim_B//2 + 1 + 2*i] -= - 1.j * np.tan(t/2)*2/T *np.exp(-1.j * mu_up)
        exponent[dim_B//2 + 1 + 2*i, dim_B//2 + 2 + 2*i] += 1 *np.cos(t) * np.exp(-1.j * mu_up)
        exponent[3*dim_B + dim_B//2 + 1 + 2*i, 3*dim_B + dim_B//2 + 2 + 2*i] += 1 *np.cos(t) * np.exp(-1.j * mu_down)

        # backward Transpose (antisymm)
        exponent[3*dim_B + dim_B//2 + 2 + 2*i, dim_B//2 + 1 + 2*i] += + 1.j * np.tan(t/2)*2/T *np.exp(-1.j * mu_down)
        exponent[3*dim_B + dim_B//2 + 1 + 2*i, dim_B//2 + 2 + 2*i] -= + 1.j * np.tan(t/2)*2/T *np.exp(-1.j * mu_up)
        exponent[dim_B//2 + 2 + 2*i, dim_B//2 + 1 + 2*i] += -1 *np.cos(t) * np.exp(-1.j * mu_up)
        exponent[3*dim_B + dim_B//2 + 2 + 2*i, 3*dim_B + dim_B//2 + 1 + 2*i] += -1 *np.cos(t) * np.exp(-1.j * mu_down)


    
    # Initial state impurity 

    """
    # (IT)
    #spin up
    exponent[dim_B//2 - 1, dim_B//2 ] += 1
    #Transpose (antisymm)
    exponent[dim_B//2, dim_B//2 - 1] -= 1
    #spin down
    exponent[3 * dim_B + dim_B//2 - 1, 3 * dim_B + dim_B//2] += 1
    #Transpose (antisymm)
    exponent[3 * dim_B + dim_B//2, 3 * dim_B + dim_B//2 - 1] -= 1
    """

    # (thermal with temperature beta)
    beta_up = 10
    beta_down = 10
    #spin up
    exponent[dim_B//2 - 1, dim_B//2 ] += np.exp(- beta_up)
    #Transpose (antisymm)
    exponent[dim_B//2, dim_B//2 - 1] -= np.exp(- beta_up)
    #spin down
    exponent[3 * dim_B + dim_B//2 - 1, 3 * dim_B + dim_B//2] += np.exp(- beta_down)
    #Transpose (antisymm)
    exponent[3 * dim_B + dim_B//2, 3 * dim_B + dim_B//2 - 1] -= np.exp(- beta_down)
    

    exponent_check = exponent


    #temporal boundary condition for 
    # spin down
    exponent_check[3 * dim_B + dim_B - 1, 3 * dim_B] += -1
    #Transpose (antisymm)
    exponent_check[3 * dim_B, 3 * dim_B + dim_B - 1] -= -1
    
    A = np.bmat([[exponent_check[1:dim_B-1,1:dim_B-1], exponent_check[1:dim_B-1,dim_B :2*dim_B], exponent_check[1:dim_B-1,2*dim_B +1:3*dim_B-1], exponent_check[1:dim_B-1,3*dim_B:]], 
                [exponent_check[dim_B :2*dim_B,1:dim_B-1], exponent_check[dim_B :2*dim_B,dim_B :2*dim_B], exponent_check[dim_B :2*dim_B,2*dim_B +1:3*dim_B-1], exponent_check[dim_B :2*dim_B,3*dim_B:]],
                [exponent_check[2*dim_B +1:3*dim_B-1,1:dim_B-1], exponent_check[2*dim_B +1:3*dim_B-1,dim_B :2*dim_B], exponent_check[2*dim_B +1:3*dim_B-1,2*dim_B +1:3*dim_B-1], exponent_check[2*dim_B +1:3*dim_B-1,3*dim_B:]],
                [exponent_check[3*dim_B:,1:dim_B-1], exponent_check[3*dim_B:,dim_B :2*dim_B], exponent_check[3*dim_B:,2*dim_B +1:3*dim_B-1], exponent_check[3*dim_B:,3*dim_B:]]])

   
    
    R = np.bmat([exponent_check[[0,dim_B -1, 2*dim_B, 3*dim_B -1],1:dim_B-1] , exponent_check[[0,dim_B -1, 2*dim_B, 3*dim_B -1],dim_B :2*dim_B],exponent_check[[0,dim_B -1, 2*dim_B, 3*dim_B -1],2*dim_B +1:3*dim_B-1] , exponent_check[[0,dim_B -1, 2*dim_B, 3*dim_B -1],3*dim_B:] ])
    C = np.zeros((4,4),dtype=np.complex_)
    C[0,1] = exponent_check[0,dim_B-1]
    C[0,2] = exponent_check[0,2*dim_B]
    C[0,3] = exponent_check[0,2*dim_B-1]

    C[1,2] = exponent_check[dim_B -1 ,2*dim_B]
    C[1,3] = exponent_check[dim_B -1,2*dim_B-1]

    C[2,3] = exponent_check[2*dim_B,2*dim_B-1]
    C -= C.T
    print(C)
    A_inv = linalg.inv(A)
    
    rho_exponent_evolved = 0.5*(R @ A_inv @ R.T + C)
    rho_evolved = np.zeros((2,2),dtype=np.complex_)
 
    rho_evolved[0,0] = 1
    rho_evolved[1,1] = - 2* rho_exponent_evolved[2,3] # minus sign because of sign-change-convention
    rho_evolved *= 1./((1+np.exp(-beta_up)) * (1+np.exp(-beta_down))) * np.sqrt(linalg.det(A)) 

    trace_vals.append( np.trace(rho_evolved))
    rho_eigvals = linalg.eigvals(rho_evolved)
    rho_eigvals_max.append(np.max(rho_eigvals))
    rho_eigvals_min.append(np.min(rho_eigvals))


    
    # temporal boundary condition for measure
    # sign because substituted in such a way that all kernels are the same.
    #spin up
    exponent[dim_B - 1, 0] += -1
    #Transpose (antisymm)
    exponent[0, dim_B - 1] -= -1
    #spin down
    exponent[3 * dim_B + dim_B - 1, 3 * dim_B] += -1
    #Transpose (antisymm)
    exponent[3 * dim_B, 3 * dim_B + dim_B - 1] -= -1
    
    exponent_inv = linalg.inv(exponent)

    with h5py.File(filename+'_spinfulpropag' + ".hdf5", 'a') as f:
        propag_data = f['propag_IM']
        #propag_data[iter] = exponent_inv[2*dim_B + 1,dim_B//2-1]
        propag_data[iter] = exponent_inv[2*dim_B + 1,3*dim_B-2] * 1./((1+np.exp(-beta_up)) * (1+np.exp(-beta_down))) 

    print(exponent_inv[2*dim_B + 1, dim_B//2-1])
    
plt.plot(np.arange(1,28)*delta_t, trace_vals[:27],linewidth=2,label='Tr'+r'$(\rho)$')
plt.plot(np.arange(1,28)*delta_t, rho_eigvals_max[:27],linewidth=2,label='max. eigenvalue of '+ r'$\rho$')
plt.plot(np.arange(1,28)*delta_t, rho_eigvals_min[:27],linewidth=2,label='min. eigenvalue of '+ r'$\rho$')
#plt.plot(np.arange(2,28)*delta_t, trace_vals_const[1:27],linewidth=2,linestyle = '--',label='fixed IM at ' + r'$T=29$')
plt.plot(np.arange(1,28)*delta_t, 27*[1.0], linestyle='--',color="grey")
plt.plot(np.arange(1,28)*delta_t, 1+np.arange(1,28)*delta_t**2*1.4, linestyle='--',color="blue")
plt.xlabel('physical time '+r'$t$')
#plt.ylabel('Tr'+r'$(\rho)$')
plt.text(.1,0.6,r'$ \delta t = {},\, \rho_\sigma(0) = \exp (-\beta_\sigma c^\dagger_\sigma c_\sigma ),\,\beta_\uparrow = {}\,\beta_\downarrow = {}$'.format(delta_t,beta_up,beta_down)+ '\n spinfull fermions,\n random spin-hopping interaction and random phases')
#plt.ylim([0.7,1.3])
plt.legend()
plt.savefig('/Users/julianthoenniss/Documents/PhD/data/'+ 'deltat='+str(delta_t) + '_betaup=' +str(beta_up)+ '_betadown=' +str(beta_down) + '_randomdynamics.pdf')