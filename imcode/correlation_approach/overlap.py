
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
def zero_to_nan(values):
    """Replace every 0 with 'nan' and return a copy."""
    return [float('nan') if x==0 else x for x in values]


np.set_printoptions(threshold=sys.maxsize, precision=3)
#filename = '/Users/julianthoenniss/Documents/PhD/data/compmode=G_o=1_Jx=0.1_Jy=0.1_g=0.0mu=0.0_del_t=1.0_beta=0.0_L=41_init=2'
#filename = '/Users/julianthoenniss/Documents/PhD/data/interleaved_Jx=0.1_Jy=0.1_g=0.0mu=0.0_del_t=1.0_L=200InfTemp-FermiSea_my_conv'
#filename = '/Users/julianthoenniss/Documents/PhD/data/corr_mich/0707/Millis_mu=0_timestep=0.01_T=100'
#filename = '/Users/julianthoenniss/Documents/PhD/data/Millis_mu0_timestep0.01_T100'
#filename = '/Users/julianthoenniss/Documents/PhD/data/XX_deltamu=0.2'
#filename = '/Users/julianthoenniss/Documents/PhD/data/corr_mich/1407/Millis_interleaved_hyb=0.05_test_T=400_deltat=0.1'
#filename = '/Users/julianthoenniss/Documents/PhD/data/Millis_interleaved_hyb=0.05_test_T=20'
filename = '/Users/julianthoenniss/Documents/PhD/data/Millis_interleaved_timestep=0.05_hyb=0.05_T=200_D=1'
#filename = '/Users/julianthoenniss/Documents/PhD/data/compmode=C_o=1_Jx=1.0_Jy=1.0_g=0.0mu=0.0_del_t=0.1_beta=0.0_L=8_init=3'
#filename = '/Users/julianthoenniss/Documents/PhD/data/Millis_mu=.2_timestep=0.5_T=30'
#filename = '/Users/julianthoenniss/Documents/PhD/data/Millis_mu=-0.2_timestep=0.05_T=50_left'
#filename = '/Users/julianthoenniss/Documents/PhD/data/compmode=C_Jx=0.1_Jy=0.1_g=0.0mu=0.0_del_t=1.0_beta=0.0_L=200_init=3'

conv = 'M'

if conv == 'J':
    #filename += '_my_conv' 
    print('using Js convention')
elif conv == 'M':
    filename += '_Michaels_conv' 
    print('using Ms convention')


max_time = 250
interval = 1
Gamma = 0.05
delta_t = 0.05
t = 0*1.1 * delta_t # hopping between spin species, factor 2 to match Michael's spin convention


with h5py.File(filename+'_DMs' + ".hdf5", 'w') as f:
    dset_DM = f.create_dataset('density_matrix', (max_time,4,4), dtype=np.complex_)

with h5py.File(filename+'_spinfulpropag' + ".hdf5", 'w') as f:
    dset_propag_IM = f.create_dataset('propag_IM', (max_time,), dtype=np.complex_)
    dset_propag_exact = f.create_dataset(
        'propag_exact', (max_time,), dtype=np.complex_)

    dset_propag_exact = f.create_dataset(
        'propag_times', (max_time,), dtype=int)

trace_vals = np.zeros(max_time,dtype=np.complex_)
trace_vals_const = np.zeros(max_time,dtype=np.complex_)
rho_eigvals_min = np.zeros(max_time,dtype=np.complex_)
rho_eigvals_max = np.zeros(max_time,dtype=np.complex_)
rho_00 = np.zeros(max_time,dtype=np.complex_)
rho_11 = np.zeros(max_time,dtype=np.complex_)
rho_22 = np.zeros(max_time,dtype=np.complex_)
rho_33 = np.zeros(max_time,dtype=np.complex_)
propag = np.zeros(max_time,dtype=np.complex_)
times = np.zeros(max_time,dtype=np.complex_)

for iter in range(1,2,interval):

    iter_readout =  0#iter
    with h5py.File(filename + '.hdf5', 'r') as f:
        #times_read = f['temp_entr']
        times_read = f['times']
        nbr_Floquet_layers= int(times_read[iter_readout])
        print('times: ', nbr_Floquet_layers)

    times[iter] = nbr_Floquet_layers
    intermediate_time = nbr_Floquet_layers
    total_time = nbr_Floquet_layers
   

    #KIC:
    norm_IM = pow(np.cos(0.1),4*(nbr_Floquet_layers))

    #XY:
    norm_IM = pow(np.cos(0.2),2*(nbr_Floquet_layers))

    #Millis:
    norm_IM=1

    B = np.zeros((4*nbr_Floquet_layers, 4*nbr_Floquet_layers),
                 dtype=np.complex_)
    with h5py.File(filename + '.hdf5', 'r') as f:
        print(4*nbr_Floquet_layers, 4*nbr_Floquet_layers)
        B = f['IM_exponent'][iter_readout, :4*nbr_Floquet_layers, :4*nbr_Floquet_layers]

    dim_B = B.shape[0]
    print('dim_b',dim_B)
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


    
    #here, the matrix B is in the Grassmann convention
  
    # adjust signs that make the influence matrix a vectorized state
    for i in range (dim_B):
        for j in range (dim_B):
            if (i+j)%2 == 1:
                B[i,j] *= -1

    print(B[dim_B-12:,dim_B-12:])
    print(linalg.inv(B)[dim_B-12:,dim_B-12:])


    exponent = np.zeros((4*dim_B, 4*dim_B), dtype=np.complex_)#this exponent will contain the exponents of both spin species as well as the impurity dynamics
    # Influence matices for both spin species
    #spin down
    #exponent[dim_B:2*dim_B, dim_B:2*dim_B] = B[:, :]
    #spin up
    exponent[2*dim_B:3*dim_B, 2*dim_B:3*dim_B] = B[:, :]

    # integration measure
    # spin up
    exponent[:dim_B, 2*dim_B :3*dim_B] = np.identity(dim_B)
    exponent[2*dim_B :3*dim_B, :dim_B] = -np.identity(dim_B)
    # spin down
    exponent[dim_B:2*dim_B, 3*dim_B:4*dim_B ] = np.identity(dim_B)
    exponent[3*dim_B:4*dim_B , dim_B:2*dim_B] = -np.identity(dim_B)
    """
    exponent[1:dim_B-1, 2*dim_B +1:3*dim_B-1] = np.identity(dim_B-2)
    exponent[2*dim_B+1 :3*dim_B-1, 1:dim_B-1] = -np.identity(dim_B-2)
    # spin down
    exponent[1+dim_B:2*dim_B-1, 1+3*dim_B:4*dim_B -1] = np.identity(dim_B-2)
    exponent[1+3*dim_B:4*dim_B -1, 1+dim_B:2*dim_B-1] = -np.identity(dim_B-2)
    """
    
    # impurity
    #hopping between spin species (gate is easily found by taking kernel of xy gate at isotropic parameters):
    seed(10)
    for i in range(dim_B//4-1):
        #t =  0.57*np.cos(0.42 * i)
        mu_up =0.3*delta_t# 0.3*np.sin(2.2 * i)
        mu_down =0*delta_t# 0.18*np.sin(1.82 * i)
        #mu_up =0# random()
        #mu_down =0# random()
        #print(t,mu_up,mu_down)

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
    beta_up = -10#-10
    beta_down = 0#-10
    #spin up
    exponent[dim_B//2 - 1, dim_B//2 ] += np.exp(- beta_up)
    #Transpose (antisymm)
    exponent[dim_B//2, dim_B//2 - 1] -= np.exp(- beta_up)
    #spin down
    exponent[3 * dim_B + dim_B//2 - 1, 3 * dim_B + dim_B//2] += np.exp(- beta_down)
    #Transpose (antisymm)
    exponent[3 * dim_B + dim_B//2, 3 * dim_B + dim_B//2 - 1] -= np.exp(- beta_down)
    
    
    exponent_check = exponent

   
    #for spin up/down density matrix --ONE TOTAL TIME
    
    delt = 2 * (total_time - intermediate_time)
    """
    A = np.bmat([[exponent_check[(delt+1):dim_B -(delt+1),(delt+1):dim_B-(delt+1)], exponent_check[(delt+1):dim_B-(delt+1),dim_B+(delt+1) :2*dim_B-(delt+1)], exponent_check[(delt+1):dim_B-(delt+1),2*dim_B +(delt+1):3*dim_B-(delt+1)], exponent_check[(delt+1):dim_B-(delt+1),3*dim_B+(delt+1):4*dim_B-(delt+1)]], 
                [exponent_check[dim_B+(delt+1) :2*dim_B-(delt+1),(delt+1):dim_B-(delt+1)], exponent_check[dim_B+(delt+1) :2*dim_B-(delt+1),dim_B+(delt+1) :2*dim_B-(delt+1)], exponent_check[dim_B+(delt+1) :2*dim_B-(delt+1),2*dim_B +(delt+1):3*dim_B-(delt+1)], exponent_check[dim_B+(delt+1) :2*dim_B-(delt+1),3*dim_B+(delt+1):4*dim_B-(delt+1)]],
                [exponent_check[2*dim_B +(delt+1):3*dim_B-(delt+1),(delt+1):dim_B-(delt+1)], exponent_check[2*dim_B +(delt+1):3*dim_B-(delt+1),dim_B+(delt+1) :2*dim_B-(delt+1)], exponent_check[2*dim_B +(delt+1):3*dim_B-(delt+1),2*dim_B +(delt+1):3*dim_B-(delt+1)], exponent_check[2*dim_B +(delt+1):3*dim_B-(delt+1),3*dim_B+(delt+1):4*dim_B-(delt+1)]],
                [exponent_check[3*dim_B+(delt+1):4*dim_B-(delt+1),(delt+1):dim_B-(delt+1)], exponent_check[3*dim_B+(delt+1):4*dim_B-(delt+1),dim_B+(delt+1) :2*dim_B-(delt+1)], exponent_check[3*dim_B+(delt+1):4*dim_B-(delt+1),2*dim_B +(delt+1):3*dim_B-(delt+1)], exponent_check[3*dim_B+(delt+1):4*dim_B-(delt+1),3*dim_B+(delt+1):4*dim_B-(delt+1)]]])


    R = np.bmat([[exponent_check[:delt+1, (delt+1):dim_B -(delt+1) ],exponent_check[:delt+1, dim_B+(delt+1) :2*dim_B-(delt+1)],exponent_check[:delt+1, 2*dim_B+(delt+1) :3*dim_B-(delt+1)],exponent_check[:delt+1, 3*dim_B+(delt+1) :4*dim_B-(delt+1)]],
                [exponent_check[dim_B-(delt+1) :dim_B+delt+1, (delt+1):dim_B -(delt+1) ],exponent_check[dim_B-(delt+1) :dim_B+delt+1,  dim_B+(delt+1) :2*dim_B-(delt+1)],exponent_check[dim_B-(delt+1) :dim_B+delt+1, 2*dim_B+(delt+1) :3*dim_B-(delt+1)],exponent_check[dim_B-(delt+1) :dim_B+delt+1, 3*dim_B+(delt+1) :4*dim_B-(delt+1)]],
                [exponent_check[2*dim_B -(delt+1):2*dim_B+delt+1, (delt+1):dim_B -(delt+1) ],exponent_check[2*dim_B -(delt+1):2*dim_B+delt+1,  dim_B+(delt+1) :2*dim_B-(delt+1)],exponent_check[2*dim_B -(delt+1):2*dim_B+delt+1, 2*dim_B+(delt+1) :3*dim_B-(delt+1)],exponent_check[2*dim_B -(delt+1):2*dim_B+delt+1, 3*dim_B+(delt+1) :4*dim_B-(delt+1)]],
                [exponent_check[3*dim_B -(delt+1):3*dim_B+delt+1, (delt+1):dim_B -(delt+1) ],exponent_check[3*dim_B -(delt+1):3*dim_B+delt+1,  dim_B+(delt+1) :2*dim_B-(delt+1)],exponent_check[3*dim_B -(delt+1):3*dim_B+delt+1,2*dim_B+(delt+1) :3*dim_B-(delt+1)],exponent_check[3*dim_B -(delt+1):3*dim_B+delt+1, 3*dim_B+(delt+1) :4*dim_B-(delt+1)]],
                [exponent_check[4*dim_B -(delt+1):4*dim_B, (delt+1):dim_B -(delt+1) ],exponent_check[4*dim_B -(delt+1):4*dim_B,  dim_B+(delt+1) :2*dim_B-(delt+1)],exponent_check[4*dim_B -(delt+1):4*dim_B, 2*dim_B+(delt+1) :3*dim_B-(delt+1)],exponent_check[4*dim_B -(delt+1):4*dim_B, 3*dim_B+(delt+1) :4*dim_B-(delt+1)]]])

    C = np.zeros((8 *(1+delt),8*(1+delt)),dtype=np.complex_)

    C[:delt+1,delt+1:2*(delt+1)] = exponent_check[:delt+1,dim_B-(delt+1) :dim_B]
    C[:delt+1,2*(delt+1):3*(delt+1)] = exponent_check[:delt+1,dim_B:dim_B+delt+1]
    C[:delt+1,3*(delt+1):4*(delt+1)] = exponent_check[:delt+1,2*dim_B -(delt+1):2*dim_B]
    C[:delt+1,4*(delt+1):5*(delt+1)] = exponent_check[:delt+1,2*dim_B:2*dim_B+delt+1]
    C[:delt+1,5*(delt+1):6*(delt+1)] = exponent_check[:delt+1,3*dim_B -(delt+1):3*dim_B]
    C[:delt+1,6*(delt+1):7*(delt+1)] = exponent_check[:delt+1,3*dim_B:3*dim_B+delt+1]
    C[:delt+1,7*(delt+1):8*(delt+1)] = exponent_check[:delt+1,4*dim_B -(delt+1):4*dim_B]

    C[1*(delt+1):2*(delt+1),2*(delt+1):3*(delt+1)] = exponent_check[dim_B-(delt+1) :dim_B,dim_B:dim_B+delt+1]
    C[1*(delt+1):2*(delt+1),3*(delt+1):4*(delt+1)] = exponent_check[dim_B-(delt+1) :dim_B,2*dim_B -(delt+1):2*dim_B]
    C[1*(delt+1):2*(delt+1),4*(delt+1):5*(delt+1)] = exponent_check[dim_B-(delt+1) :dim_B,2*dim_B:2*dim_B+delt+1]
    C[1*(delt+1):2*(delt+1),5*(delt+1):6*(delt+1)] = exponent_check[dim_B-(delt+1) :dim_B,3*dim_B -(delt+1):3*dim_B]
    C[1*(delt+1):2*(delt+1),6*(delt+1):7*(delt+1)] = exponent_check[dim_B-(delt+1) :dim_B,3*dim_B:3*dim_B+delt+1]
    C[1*(delt+1):2*(delt+1),7*(delt+1):8*(delt+1)] = exponent_check[dim_B-(delt+1) :dim_B,4*dim_B -(delt+1):4*dim_B]

    C[2*(delt+1):3*(delt+1),3*(delt+1):4*(delt+1)] = exponent_check[dim_B:dim_B+delt+1,2*dim_B -(delt+1):2*dim_B]
    C[2*(delt+1):3*(delt+1),4*(delt+1):5*(delt+1)] = exponent_check[dim_B:dim_B+delt+1,2*dim_B:2*dim_B+delt+1]
    C[2*(delt+1):3*(delt+1),5*(delt+1):6*(delt+1)] = exponent_check[dim_B:dim_B+delt+1,3*dim_B -(delt+1):3*dim_B]
    C[2*(delt+1):3*(delt+1),6*(delt+1):7*(delt+1)] = exponent_check[dim_B:dim_B+delt+1,3*dim_B:3*dim_B+delt+1]
    C[2*(delt+1):3*(delt+1),7*(delt+1):8*(delt+1)] = exponent_check[dim_B:dim_B+delt+1,4*dim_B -(delt+1):4*dim_B]

    C[3*(delt+1):4*(delt+1),4*(delt+1):5*(delt+1)] = exponent_check[2*dim_B -(delt+1):2*dim_B,2*dim_B+delt+1]
    C[3*(delt+1):4*(delt+1),5*(delt+1):6*(delt+1)] = exponent_check[2*dim_B -(delt+1):2*dim_B,3*dim_B -(delt+1):3*dim_B]
    C[3*(delt+1):4*(delt+1),6*(delt+1):7*(delt+1)] = exponent_check[2*dim_B -(delt+1):2*dim_B,3*dim_B:3*dim_B+delt+1]
    C[3*(delt+1):4*(delt+1),7*(delt+1):8*(delt+1)] = exponent_check[2*dim_B -(delt+1):2*dim_B,4*dim_B -(delt+1):4*dim_B]

    C[4*(delt+1):5*(delt+1),5*(delt+1):6*(delt+1)] = exponent_check[2*dim_B:2*dim_B+delt+1,3*dim_B -(delt+1):3*dim_B]
    C[4*(delt+1):5*(delt+1),6*(delt+1):7*(delt+1)] = exponent_check[2*dim_B:2*dim_B+delt+1,3*dim_B:3*dim_B+delt+1]
    C[4*(delt+1):5*(delt+1),7*(delt+1):8*(delt+1)] = exponent_check[2*dim_B:2*dim_B+delt+1,4*dim_B -(delt+1):4*dim_B]

    C[5*(delt+1):6*(delt+1),6*(delt+1):7*(delt+1)] = exponent_check[3*dim_B -(delt+1):3*dim_B,3*dim_B:3*dim_B+delt+1]
    C[5*(delt+1):6*(delt+1),7*(delt+1):8*(delt+1)] = exponent_check[3*dim_B -(delt+1):3*dim_B,4*dim_B -(delt+1):4*dim_B]

    C[6*(delt+1):7*(delt+1),7*(delt+1):8*(delt+1)] = exponent_check[3*dim_B:3*dim_B+delt+1,4*dim_B -(delt+1):4*dim_B]

    C -= C.T

    A_inv = linalg.inv(A)
    
    rho_exponent_evolved = 0.5*(R @ A_inv @ R.T + C)

    rho_evolved = np.zeros((4,4),dtype=np.complex_)

    # minus signs because of sign-change-convention, factor 2 bc of antisymmetry
    a1 =  - 2 * rho_exponent_evolved[2*(1+delt) +delt ,2*(1+delt) +delt+1]
    a2 = +2 * rho_exponent_evolved[2*(1+delt) +delt,4*(1+delt) +delt]
    a3 = -2 * rho_exponent_evolved[2*(1+delt) +delt,4*(1+delt) +delt+1]
    a4 = -2 * rho_exponent_evolved[2*(1+delt) +delt+1,4*(1+delt) +delt]
    a5 = +2 * rho_exponent_evolved[2*(1+delt) +delt+1,4*(1+delt) +delt+1]
    a6 = -2 * rho_exponent_evolved[4*(1+delt) +delt,4*(1+delt) +delt+1]

    rho_evolved[0,0] = 1
    rho_evolved[0,3] = - a5

    rho_evolved[1,1] = a6
    rho_evolved[1,2] = - a4

    rho_evolved[2,1] = a3
    rho_evolved[2,2] = a1

    rho_evolved[3,0] = a2
    rho_evolved[3,3] = a1*a6 - a2*a5 + a3*a4

    rho_evolved *= 1./((1+np.exp(-beta_up)))  * np.sqrt(linalg.det(A)) * norm_IM #* 1./(1+np.exp(-beta_down))

    trace_vals[iter]=( np.trace(rho_evolved))
    rho_eigvals = linalg.eigvals(rho_evolved)
    rho_eigvals_max[iter]=(np.max(rho_eigvals))
    rho_eigvals_min[iter]=(np.min(rho_eigvals))
    rho_00[iter]=(np.real(rho_evolved[0,0]))
    rho_11[iter]=(np.real(rho_evolved[1,1]))
    rho_22[iter]=(np.real(rho_evolved[2,2]))
    rho_33[iter]=(np.real(rho_evolved[3,3]))

    """

    """
    #for spin up/down density matrix

    A = np.bmat([[exponent_check[1:dim_B-1,1:dim_B-1], exponent_check[1:dim_B-1,dim_B+1 :2*dim_B-1], exponent_check[1:dim_B-1,2*dim_B +1:3*dim_B-1], exponent_check[1:dim_B-1,3*dim_B+1:4*dim_B-1]], 
                [exponent_check[dim_B+1 :2*dim_B-1,1:dim_B-1], exponent_check[dim_B+1 :2*dim_B-1,dim_B+1 :2*dim_B-1], exponent_check[dim_B+1 :2*dim_B-1,2*dim_B +1:3*dim_B-1], exponent_check[dim_B+1 :2*dim_B-1,3*dim_B+1:4*dim_B-1]],
                [exponent_check[2*dim_B +1:3*dim_B-1,1:dim_B-1], exponent_check[2*dim_B +1:3*dim_B-1,dim_B+1 :2*dim_B-1], exponent_check[2*dim_B +1:3*dim_B-1,2*dim_B +1:3*dim_B-1], exponent_check[2*dim_B +1:3*dim_B-1,3*dim_B+1:4*dim_B-1]],
                [exponent_check[3*dim_B+1:4*dim_B-1,1:dim_B-1], exponent_check[3*dim_B+1:4*dim_B-1,dim_B+1 :2*dim_B-1], exponent_check[3*dim_B+1:4*dim_B-1,2*dim_B +1:3*dim_B-1], exponent_check[3*dim_B+1:4*dim_B-1,3*dim_B+1:4*dim_B-1]]])

   
    
    R = np.bmat([exponent_check[[0,dim_B -1,dim_B,2*dim_B -1, 2*dim_B, 3*dim_B -1,3*dim_B, 4*dim_B -1],1:dim_B-1] ,exponent_check[[0,dim_B -1,dim_B,2*dim_B -1, 2*dim_B, 3*dim_B -1,3*dim_B, 4*dim_B -1],dim_B+1:2*dim_B-1],exponent_check[[0,dim_B -1,dim_B,2*dim_B -1, 2*dim_B, 3*dim_B -1,3*dim_B, 4*dim_B -1],2*dim_B+1:3*dim_B-1],exponent_check[[0,dim_B -1,dim_B,2*dim_B -1, 2*dim_B, 3*dim_B -1,3*dim_B, 4*dim_B -1],3*dim_B+1:4*dim_B-1] ])
    C = np.zeros((8,8),dtype=np.complex_)
    C[0,1] = exponent_check[0,dim_B-1]
    C[0,2] = exponent_check[0,dim_B]
    C[0,3] = exponent_check[0,2*dim_B-1]
    C[0,4] = exponent_check[0,2*dim_B]
    C[0,5] = exponent_check[0,3*dim_B-1]
    C[0,6] = exponent_check[0,3*dim_B]
    C[0,7] = exponent_check[0,4*dim_B-1]

    C[1,2] = exponent_check[dim_B-1,dim_B]
    C[1,3] = exponent_check[dim_B-1,2*dim_B-1]
    C[1,4] = exponent_check[dim_B-1,2*dim_B]
    C[1,5] = exponent_check[dim_B-1,3*dim_B-1]
    C[1,6] = exponent_check[dim_B-1,3*dim_B]
    C[1,7] = exponent_check[dim_B-1,4*dim_B-1]

    C[2,3] = exponent_check[dim_B,2*dim_B-1]
    C[2,4] = exponent_check[dim_B,2*dim_B]
    C[2,5] = exponent_check[dim_B,3*dim_B-1]
    C[2,6] = exponent_check[dim_B,3*dim_B]
    C[2,7] = exponent_check[dim_B,4*dim_B-1]

    C[3,4] = exponent_check[2*dim_B-1,2*dim_B]
    C[3,5] = exponent_check[2*dim_B-1,3*dim_B-1]
    C[3,6] = exponent_check[2*dim_B-1,3*dim_B]
    C[3,7] = exponent_check[2*dim_B-1,4*dim_B-1]

    C[4,5] = exponent_check[2*dim_B,3*dim_B-1]
    C[4,6] = exponent_check[2*dim_B,3*dim_B]
    C[4,7] = exponent_check[2*dim_B,4*dim_B-1]

    C[5,6] = exponent_check[3*dim_B-1,3*dim_B]
    C[5,7] = exponent_check[3*dim_B-1,4*dim_B-1]

    C[6,7] = exponent_check[3*dim_B,4*dim_B-1]

    C -= C.T

    A_inv = linalg.inv(A)
    
    rho_exponent_evolved = 0.5*(R @ A_inv @ R.T + C)

    rho_evolved = np.zeros((4,4),dtype=np.complex_)

    # minus signs because of sign-change-convention, factor 2 bc of antisymmetry
    a1 =  - 2 * rho_exponent_evolved[2,3]
    a2 = +2 * rho_exponent_evolved[2,4]
    a3 = -2 * rho_exponent_evolved[2,5]
    a4 = -2 * rho_exponent_evolved[3,4]
    a5 = +2 * rho_exponent_evolved[3,5]
    a6 = -2 * rho_exponent_evolved[4,5]

    rho_evolved[0,0] = 1
    rho_evolved[0,3] = - a5

    rho_evolved[1,1] = a6
    rho_evolved[1,2] = - a4

    rho_evolved[2,1] = a3
    rho_evolved[2,2] = a1

    rho_evolved[3,0] = a2
    rho_evolved[3,3] = a1*a6 - a2*a5 + a3*a4

    rho_evolved *= 1./((1+np.exp(-beta_up))) * 1./(1+np.exp(-beta_down)) * np.sqrt(linalg.det(A)) * norm_IM

    trace_vals[iter]=( np.trace(rho_evolved))
    rho_eigvals = linalg.eigvals(rho_evolved)
    rho_eigvals_max[iter]=(np.max(rho_eigvals))
    rho_eigvals_min[iter]=(np.min(rho_eigvals))
    rho_00[iter]=(np.real(rho_evolved[0,0]))
    rho_11[iter]=(np.real(rho_evolved[1,1]))
    rho_22[iter]=(np.real(rho_evolved[2,2]))
    rho_33[iter]=(np.real(rho_evolved[3,3]))
    """
   
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
    #print('Z', np.sqrt(linalg.det(exponent)))
    for iter in range (2,200):
        #times[iter] = iter
        delt = 2 * (total_time - iter)
        with h5py.File(filename+'_spinfulpropag' + ".hdf5", 'a') as f:
            propag_data = f['propag_IM']
            #propag_data[iter] = exponent_inv[2*dim_B + 1,dim_B//2-1]
            #propag_data[iter] = exponent_inv[2*dim_B + 1,3*dim_B-2] #this is the propagator for the spin ups
            #propag_data[iter] = exponent_inv[2*dim_B + delt,3*dim_B -1- delt] #<n(t)># works
            #propag_data[iter] = exponent_inv[2*dim_B + delt+1, delt+1]  #<n(t)># works, used for benchmark
            #propag_data[iter] = exponent_inv[delt+1, dim_B -1 - delt-1] # works also
            propag_data[iter] = -exponent_inv[2*dim_B + dim_B//2 -2*iter -1 , dim_B//2 -2*iter -1]
            propag[iter] = propag_data[iter]

            times_data = f['propag_times']
            times_data[iter] = iter #nbr_Floquet_layers
            times[iter] = times_data[iter]
    #with h5py.File(filename+'_DMs' + ".hdf5", 'a') as f:
    #    DM_data = f['density_matrix']
    #    DM_data[iter,:,:] = rho_evolved[:,:]
    #print(exponent_inv[2*dim_B + 1, dim_B//2-1])
    #print(times)
    #print(propag)

#plt.plot(np.arange(1,max_time,interval)*delta_t* Gamma, trace_vals[1:max_time:interval],linewidth=2,label='Tr'+r'$(\rho)$')
#plt.plot(np.arange(1,max_time,interval)*delta_t* Gamma, rho_eigvals_max[1:max_time:interval],linewidth=2,linestyle= '-',label='max. eigenvalue of '+ r'$\rho$')
#plt.plot(np.arange(1,max_time,interval)*delta_t* Gamma, rho_eigvals_min[1:max_time:interval],linewidth=2,label='min. eigenvalue of '+ r'$\rho$')
#plt.plot(np.arange(1,max_time,interval)*delta_t* Gamma, rho_00[1:max_time:interval],linewidth=2,linestyle= '--',label=r'$\rho_{00}$')
#plt.plot(np.arange(1,max_time,interval)*delta_t * Gamma, rho_11[1:max_time+1:interval],linewidth=2,linestyle= 'dotted',label=r'$\rho_{11}$')
#plt.plot(np.arange(1,max_time,interval)*delta_t* Gamma, rho_22[1:max_time+1:interval],linewidth=2,linestyle= '--',alpha=0.8,label=r'$\rho_{22}$')
plt.plot(times[1::interval]*delta_t* Gamma,propag[1::interval],linewidth=2,linestyle= '--',alpha=0.8,label='Our IM result via Grassmanns')
#plt.plot(np.arange(1,max_time,interval)*delta_t* Gamma, rho_33[1:max_time:interval],linewidth=2,linestyle= 'dotted',label=r'$\rho_{33}$')
#plt.plot(np.arange(2,28)*delta_t* Gamma, trace_vals_const[1:27],linewidth=2,linestyle = '--',label='fixed IM at ' + r'$T=29$')
#plt.plot(np.arange(1,max_time)*delta_t* Gamma, (max_time - 1) *[1.0], linestyle='--',color="grey")
#plt.plot(np.arange(1,28)*delta_t* Gamma, 1+np.arange(1,28)*delta_t**2*1.4, linestyle='--',color="blue")
Millis_IM_x = [0,0.1,0.2,0.25,0.3,0.35,0.4,0.5,0.6,0.8,1.0,1.2,1.4]
Millis_IM_y = [0,0.0485,0.0948,0.0994, 0.102,0.106, 0.1105, 0.113, 0.1166, 0.113, 0.1008, 0.092, 0.085]
plt.scatter(Millis_IM_x,Millis_IM_y,label='Millis')

plt.xlabel('rescaled physical time (as in Millis) '+r'$t \Gamma$')
plt.ylabel(r'$\langle n_{d,\sigma} \rangle$')
#plt.text(.1,0.1,r'$ \delta t = {},\, \rho_\sigma(0) = \exp (-\beta_\sigma c^\dagger_\sigma c_\sigma ),\,\beta_\uparrow = {}\,\beta_\downarrow = {}$'.format(delta_t,beta_up,beta_down)+ '\n spinfull fermions')
#plt.ylim([-1,5.1])
plt.legend()
plt.savefig('/Users/julianthoenniss/Documents/PhD/data/'+ 'deltat='+str(delta_t) + '_betaup=' +str(beta_up)+ '_betadown=' +str(beta_down) + '_randomdynamics.pdf')