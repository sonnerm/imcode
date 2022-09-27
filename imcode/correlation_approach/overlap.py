
from email import iterators
import numpy as np
from scipy import linalg
import sys
from create_correlation_block import create_correlation_block
import h5py
import matplotlib.pyplot as plt
# seed the pseudorandom number generator
from random import seed
from random import random
# seed random number generator
def zero_to_nan(values):
    """Replace every 0 with 'nan' and return a copy."""
    return [float('nan') if x==0 else x for x in values]


np.set_printoptions(threshold=sys.maxsize, precision=6)
#filename = '/Users/julianthoenniss/Documents/PhD/data/compmode=G_o=1_Jx=0.1_Jy=0.1_g=0.0mu=0.0_del_t=1.0_beta=0.0_L=41_init=2'
#filename = '/Users/julianthoenniss/Documents/PhD/data/interleaved_Jx=0.1_Jy=0.1_g=0.0mu=0.0_del_t=1.0_L=200InfTemp-FermiSea_my_conv'
#filename = '/Users/julianthoenniss/Documents/PhD/data/corr_mich/0707/Millis_mu=0_timestep=0.01_T=100'
#filename = '/Users/julianthoenniss/Documents/PhD/data/Millis_mu0_timestep0.01_T100'
#filename = '/Users/julianthoenniss/Documents/PhD/data/XX_deltamu=0.2'
#filename = '/Users/julianthoenniss/Documents/PhD/data/Millis_interleaved_timestep=0.1_hyb=0.05_T=50-200_Delta=1'
#filename = '/Users/julianthoenniss/Documents/PhD/data/Millis_interleaved_timestep=0.075_hyb=0.05_T=50-200_Delta=1'
#filename = '/Users/julianthoenniss/Documents/PhD/data/Millis_interleaved_hyb=0.05_test_T=20'
#filename = '/Users/julianthoenniss/Documents/PhD/data/Millis_interleaved_hyb=0.05_test_T=400_deltat=0.035_Delta=1'
#filename = '/Users/julianthoenniss/Documents/PhD/data/compmode=C_o=1_Jx=1.0_Jy=1.0_g=0.0mu=0.0_del_t=0.1_beta=0.0_L=8_init=3'
#filename = '/Users/julianthoenniss/Documents/PhD/data/Millis_mu=.2_timestep=0.5_T=30'
filename = '/Users/julianthoenniss/Documents/PhD/data/corr_mich/0909/Cohen/Cohen2015_inchworm_deltat=0.01_shorttime_doublhyb_300-1200'
#filename = '/Users/julianthoenniss/Documents/PhD/data/corr_mich/0909/Arrigoni/Arrigoni_mode=C_o=1_Jx=0.5_Jy=0.5_g=0.0mu=0.0_del_t=-0.1_beta=0.0_L=1200_init=2_coupling=0.3162'
#filename = '/Users/julianthoenniss/Documents/PhD/data/XX_deltamu=0.2_beta=50.0_deltat=-0.1_coupling=0.5'

conv = 'M'
mode = 1 #1: one full-time IM where legs above evolution time are contracted and intergated, 2: for each evolution time, use individual IM, 3: compute propagator, not density matrix (automatically normalized)

if conv == 'J':
    #filename += '_my_conv' 
    print('using Js convention')
elif conv == 'M':
    filename += '_Michaels_conv' 
    print('using Ms convention')


max_time = 900
interval = 1
Gamma = 1.
delta_t = 0.01
t = 0. * delta_t#-3*0.5*delta_t#-100#4 * delta_t # hopping between spin species, factor 2 to match Michael's spin convention



with h5py.File(filename+'_spinfulpropag' + ".hdf5", 'w') as f:
    dset_propag_IM = f.create_dataset('propag_IM', (max_time,), dtype=np.clongdouble)
    dset_propag_exact = f.create_dataset(
        'propag_exact', (max_time,), dtype=np.clongdouble)
    dset_propag_exact = f.create_dataset('rho_00', (max_time,), dtype=np.clongdouble)
    dset_propag_exact = f.create_dataset('rho_11', (max_time,), dtype=np.clongdouble)
    dset_propag_exact = f.create_dataset('rho_22', (max_time,), dtype=np.clongdouble)
    dset_propag_exact = f.create_dataset('rho_33', (max_time,), dtype=np.clongdouble)
    dset_propag_exact = f.create_dataset('trace', (max_time,), dtype=np.clongdouble)
    dset_propag_exact = f.create_dataset('max_eigvals', (max_time,), dtype=np.clongdouble)
    dset_propag_exact = f.create_dataset('min_eigvals', (max_time,), dtype=np.clongdouble)
    dset_propag_exact = f.create_dataset('rho_full', (max_time,4,4), dtype=np.clongdouble)
  
    dset_propag_exact = f.create_dataset(
        'propag_times', (max_time,), dtype=int)

trace_vals = np.zeros(max_time,dtype=np.clongdouble)
trace_vals_const = np.zeros(max_time,dtype=np.clongdouble)
rho_eigvals_min = np.zeros(max_time,dtype=np.clongdouble)
rho_eigvals_max = np.zeros(max_time,dtype=np.clongdouble)
rho_00 = np.zeros(max_time,dtype=np.clongdouble)
rho_11 = np.zeros(max_time,dtype=np.clongdouble)
rho_22 = np.zeros(max_time,dtype=np.clongdouble)
rho_33 = np.zeros(max_time,dtype=np.clongdouble)
rho_full = np.zeros((max_time,4,4),dtype=np.clongdouble)
propag = np.zeros(max_time,dtype=np.clongdouble)
times = np.zeros(max_time,dtype=np.int_)

for iter in range(0,1,interval):

    iter_readout = iter
    with h5py.File(filename + '.hdf5', 'r') as f:

        #read out time from chain:
        #times_read = f['temp_entr']
        #nbr_Floquet_layers  = int(times_read[iter,0])

        #read out time from specdens
        times_read = f['times']
        nbr_Floquet_layers= int(times_read[iter_readout])#300
        #nbr_Floquet_layers = iter_readout + 1
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
                 dtype=np.clongdouble)
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
        S = np.zeros(B.shape,dtype=np.clongdouble)
        for i in range (dim_B//4):#order plus and minus next to each other
            S [dim_B // 2 - (2 * i) - 2,4 * i] = 1
            S [dim_B // 2 - (2 * i) - 1,4 * i + 2] = 1
            S [dim_B // 2 + (2 * i) ,4 * i + 1] = 1
            S [dim_B // 2 + (2 * i) + 1,4 * i + 3] = 1

        #the following two transformation bring it into in/out- basis (not theta, zeta)
        rot = np.zeros(B.shape,dtype=np.clongdouble)
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

    print(B[dim_B-6:,dim_B-6:])
    #print(linalg.inv(B)[dim_B-12:,dim_B-12:])


    exponent = np.zeros((4*dim_B, 4*dim_B), dtype=np.clongdouble)#this exponent will contain the exponents of both spin species as well as the impurity dynamics
    # Influence matices for both spin species
    #spin down
    exponent[dim_B:2*dim_B, dim_B:2*dim_B] = B[:, :]
    #spin up
    exponent[2*dim_B:3*dim_B, 2*dim_B:3*dim_B] = B[:, :]

    # integration measure
    # spin up
    exponent[:dim_B, 2*dim_B :3*dim_B] += np.identity(dim_B)
    exponent[2*dim_B :3*dim_B, :dim_B] += -np.identity(dim_B)
    # spin down
    exponent[dim_B:2*dim_B, 3*dim_B:4*dim_B ] += np.identity(dim_B)
    exponent[3*dim_B:4*dim_B , dim_B:2*dim_B] += -np.identity(dim_B)
  
    
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
    beta_up = 100#-100#-10
    beta_down = -100#100#-10
    #spin up
    exponent[dim_B//2 - 1, dim_B//2 ] += np.exp(- beta_up)
    #Transpose (antisymm)
    exponent[dim_B//2, dim_B//2 - 1] -= np.exp(- beta_up)
    #spin down
    exponent[3 * dim_B + dim_B//2 - 1, 3 * dim_B + dim_B//2] += np.exp(- beta_down)
    #Transpose (antisymm)
    exponent[3 * dim_B + dim_B//2, 3 * dim_B + dim_B//2 - 1] -= np.exp(- beta_down)
    


    
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
    
    
    if mode == 1: #one final-time IM for all evolution times
    
        for intermediate_time_dm in range (5,66,20):
            print(intermediate_time_dm)
            exponent_check = exponent.copy()

            #impurity
            #hopping between spin species (gate is easily found by taking kernel of xy gate at isotropic parameters):
            seed(10)
            for i in range(intermediate_time_dm):
                #t =  0.57*np.cos(0.42 * i)
        
                mu_up =0#0.3 * delta_t#-0.3*delta_t#0.3*delta_t# 0.3*np.sin(2.2 * i)
                mu_down =0#0.3 * delta_t#-0.3*delta_t#0.3*delta_t# 0.18*np.sin(1.82 * i)
                #mu_up =0# random()
                #mu_down =0# random()
                #print(t,mu_up,mu_down)

                T=1+np.tan(t/2)**2
                # forward 
                # (matrix elements between up -> down)
                exponent_check[dim_B//2 - 3 - 2*i, 3*dim_B + dim_B//2 - 2 - 2*i] += 1.j * np.tan(t/2) *2/T *np.exp(1.j * mu_up)
                exponent_check[dim_B//2 - 2 - 2*i, 3*dim_B + dim_B//2 - 3 - 2*i] -= 1.j * np.tan(t/2)*2/T *np.exp(1.j * mu_down)
                #(matrix elements between up -> up)
                exponent_check[dim_B//2 - 3 - 2*i, dim_B//2 - 2 - 2*i] += 1 *np.cos(t) *np.exp(1.j * mu_up)
                exponent_check[3*dim_B + dim_B//2 - 3 - 2*i, 3*dim_B + dim_B//2 - 2 - 2*i] += 1 *np.cos(t) *np.exp(1.j * mu_down)

                # forward Transpose (antisymm)
                exponent_check[3*dim_B + dim_B//2 - 2 - 2*i, dim_B//2 - 3 - 2*i] += -1.j * np.tan(t/2)*2/T *np.exp(1.j * mu_up)
                exponent_check[3*dim_B + dim_B//2 - 3 - 2*i, dim_B//2 - 2 - 2*i] -= -1.j * np.tan(t/2)*2/T *np.exp(1.j * mu_down)
                exponent_check[dim_B//2 - 2 - 2*i,dim_B//2 - 3 - 2*i] += -1 *np.cos(t) *np.exp(1.j * mu_up)
                exponent_check[3*dim_B + dim_B//2 - 2 - 2*i, 3*dim_B + dim_B//2 - 3 - 2*i] += -1 *np.cos(t) *np.exp(1.j * mu_down)

                # backward
                exponent_check[dim_B//2 + 1 + 2*i, 3*dim_B + dim_B//2 + 2 + 2*i] += - 1.j * np.tan(t/2)*2/T *np.exp(-1.j * mu_down)
                exponent_check[dim_B//2 + 2 + 2*i, 3*dim_B + dim_B//2 + 1 + 2*i] -= - 1.j * np.tan(t/2)*2/T *np.exp(-1.j * mu_up)
                exponent_check[dim_B//2 + 1 + 2*i, dim_B//2 + 2 + 2*i] += 1 *np.cos(t) * np.exp(-1.j * mu_up)
                exponent_check[3*dim_B + dim_B//2 + 1 + 2*i, 3*dim_B + dim_B//2 + 2 + 2*i] += 1 *np.cos(t) * np.exp(-1.j * mu_down)

                # backward Transpose (antisymm)
                exponent_check[3*dim_B + dim_B//2 + 2 + 2*i, dim_B//2 + 1 + 2*i] += + 1.j * np.tan(t/2)*2/T *np.exp(-1.j * mu_down)
                exponent_check[3*dim_B + dim_B//2 + 1 + 2*i, dim_B//2 + 2 + 2*i] -= + 1.j * np.tan(t/2)*2/T *np.exp(-1.j * mu_up)
                exponent_check[dim_B//2 + 2 + 2*i, dim_B//2 + 1 + 2*i] += -1 *np.cos(t) * np.exp(-1.j * mu_up)
                exponent_check[3*dim_B + dim_B//2 + 2 + 2*i, 3*dim_B + dim_B//2 + 1 + 2*i] += -1 *np.cos(t) * np.exp(-1.j * mu_down)
            
            for i in range(intermediate_time_dm, dim_B//4-1):
                exponent_check[dim_B//2 - 3 - 2*i, dim_B//2 + 2 + 2*i] += 1 
                exponent_check[dim_B//2 - 2 - 2*i, dim_B//2 + 1 + 2*i] += 1 

                exponent_check[3*dim_B + dim_B//2 - 3 - 2*i,3*dim_B +  dim_B//2 + 2 + 2*i] += 1 
                exponent_check[3*dim_B + dim_B//2 - 2 - 2*i,3*dim_B +  dim_B//2 + 1 + 2*i] += 1 

                #antisymm. transpose
                exponent_check[dim_B//2 + 2 + 2*i, dim_B//2 - 3 - 2*i] += -1 
                exponent_check[dim_B//2 + 1 + 2*i, dim_B//2 - 2 - 2*i] += -1 

                exponent_check[3*dim_B +  dim_B//2 + 2 + 2*i, 3*dim_B + dim_B//2 - 3 - 2*i] += -1 
                exponent_check[3*dim_B +  dim_B//2 + 1 + 2*i, 3*dim_B + dim_B//2 - 2 - 2*i] += -1 
            
            delt = 2 * (total_time - intermediate_time_dm)
                        
            #take out measure connecting cut legs
            #spin up
            exponent_check[delt+1, 2*dim_B + delt+1] -= 1
            #spin down
            exponent_check[dim_B + delt+1, 3*dim_B + delt+1] -= 1
            #antisymm
            exponent_check[2*dim_B + delt+1, delt+1] += 1
            exponent_check[3*dim_B + delt+1, dim_B + delt+1] += 1

        

            A= np.zeros((4*(dim_B-2),4*(dim_B-2)),dtype=np.clongdouble)
            R= np.zeros((8,4*(dim_B-2)),dtype=np.clongdouble)
            C = np.zeros((8 ,8),dtype=np.clongdouble)


            for i in range (4):
                for j in range (4):
    
                    #for spin up/down density matrix --ONE TOTAL TIME --upper part integrated
                    A[i * (dim_B-2):i * (dim_B-2)+delt+1,j * (dim_B-2):j * (dim_B-2) + delt+1] = exponent_check[i * dim_B:i * dim_B+delt+1,j * dim_B:j * dim_B+delt+1]
                    A[i * (dim_B-2):i * (dim_B-2)+delt+1,j * (dim_B-2) +delt + 1 :j * (dim_B-2) + delt+1 +(dim_B -2*(delt+2)) ] = exponent_check[i * dim_B:i * dim_B+delt+1,j * dim_B+(delt+2) :j * dim_B+dim_B-(delt+2)]
                    A[i * (dim_B-2):i * (dim_B-2)+delt+1,j * (dim_B-2) +delt+1 +(dim_B -2*(delt+2)):j * (dim_B-2) +dim_B-2] = exponent_check[i * dim_B:i * dim_B+delt+1,j * dim_B+dim_B -(delt+1):j * dim_B+dim_B]

                    A[i * (dim_B-2) +delt + 1 :i * (dim_B-2) + delt+1 +(dim_B -2*(delt+2)) ,j * (dim_B-2):j * (dim_B-2) + delt+1 ] =  exponent_check[i * dim_B+(delt+2) :i * dim_B+dim_B-(delt+2),j * dim_B:j * dim_B+delt+1]
                    A[i * (dim_B-2) +delt + 1 :i * (dim_B-2) + delt+1 +(dim_B -2*(delt+2)) ,j * (dim_B-2) +delt + 1 :j * (dim_B-2) + delt+1 +(dim_B -2*(delt+2)) ] = exponent_check[i * dim_B+(delt+2) :i * dim_B+dim_B-(delt+2),j * dim_B+(delt+2) :j * dim_B+dim_B-(delt+2)]
                    A[i * (dim_B-2) +delt + 1 :i * (dim_B-2) + delt+1 +(dim_B -2*(delt+2)) ,j * (dim_B-2) +delt+1 +(dim_B -2*(delt+2)):j * (dim_B-2) +dim_B-2] = exponent_check[i * dim_B+(delt+2) :i * dim_B+dim_B-(delt+2),j * dim_B+dim_B -(delt+1):j * dim_B+dim_B]

                    A[i * (dim_B-2) +delt+1 +(dim_B -2*(delt+2)):i * (dim_B-2) +dim_B-2,j * (dim_B-2):j * (dim_B-2) + delt+1] =   exponent_check[i * dim_B+dim_B -(delt+1):i * dim_B+dim_B,j * dim_B:j * dim_B+delt+1]
                    A[i * (dim_B-2) +delt+1 +(dim_B -2*(delt+2)):i * (dim_B-2) +dim_B-2,j * (dim_B-2) +delt + 1 :j * (dim_B-2) + delt+1 +(dim_B -2*(delt+2))] = exponent_check[i * dim_B+dim_B -(delt+1):i * dim_B+dim_B,j * dim_B+(delt+2) :j * dim_B+dim_B-(delt+2)]
                    A[i * (dim_B-2) +delt+1 +(dim_B -2*(delt+2)):i * (dim_B-2) +dim_B-2,j * (dim_B-2) +delt+1 +(dim_B -2*(delt+2)):j * (dim_B-2) +dim_B-2] =  exponent_check[i * dim_B+dim_B -(delt+1):i * dim_B+dim_B,j * dim_B+dim_B -(delt+1):j * dim_B+dim_B]


                    C[2*i,2*j] = exponent_check[i*dim_B + delt+1,j * dim_B + delt+1]
                    C[2*i,2*j+1] = exponent_check[i*dim_B + delt+1,j * dim_B + dim_B-(delt+2)]
                    C[2*i+1,2*j] = exponent_check[i * dim_B + dim_B-(delt+2),j*dim_B + delt+1]
                    C[2*i+1,2*j+1] = exponent_check[i*dim_B + dim_B-(delt+2),j * dim_B +dim_B-(delt+2)]
                    

                    R[2*i,j * (dim_B-2):j * (dim_B-2) + delt+1] = exponent_check[i * dim_B+delt+1,j * dim_B:j * dim_B+delt+1]
                    R[2*i,j * (dim_B-2) +delt + 1 :j * (dim_B-2) + delt+1 +(dim_B -2*(delt+2)) ] = exponent_check[i * dim_B+delt+1,j * dim_B+(delt+2) :j * dim_B+dim_B-(delt+2)]
                    R[2*i,j * (dim_B-2) +delt+1 +(dim_B -2*(delt+2)):j * (dim_B-2) +dim_B-2 ] = exponent_check[i * dim_B+delt+1,j * dim_B+dim_B -(delt+1):j * dim_B+dim_B]

                    R[2*i+1,j * (dim_B-2):j * (dim_B-2) + delt+1] = exponent_check[i * dim_B+dim_B-(delt+2),j * dim_B:j * dim_B+delt+1]
                    R[2*i+1,j * (dim_B-2) +delt + 1 :j * (dim_B-2) + delt+1 +(dim_B -2*(delt+2)) ] = exponent_check[i * dim_B+dim_B-(delt+2),j * dim_B+(delt+2) :j * dim_B+dim_B-(delt+2)]
                    R[2*i+1,j * (dim_B-2) +delt+1 +(dim_B -2*(delt+2)):j * (dim_B-2) +dim_B-2 ] = exponent_check[i * dim_B+dim_B-(delt+2),j * dim_B+dim_B -(delt+1):j * dim_B+dim_B]


            A_inv = linalg.inv(A)
            
            rho_exponent_evolved = 0.5*(R @ A_inv @ R.T+C )

            
            
            #boundary condition for legs above cut
            rho_exponent_evolved *= 2

            rho_exponent_evolved [2,3] += -1
            rho_exponent_evolved [3,2] -= -1

            rho_exponent_evolved [4,5] += -1
            rho_exponent_evolved [5,4] -= -1

            rho_exponent_evolved_A = np.zeros((4,4),dtype=np.clongdouble)
            rho_exponent_evolved_R = np.zeros((4,4),dtype=np.clongdouble)
            rho_exponent_evolved_C = np.zeros((4,4),dtype=np.clongdouble)
            
            rho_exponent_evolved_A[:2,:2] = 0.5 * rho_exponent_evolved[2:4,2:4]#factor 1/2 to avoid double counting in antisymmetrization
            rho_exponent_evolved_A[2:4,2:4] = 0.5 * rho_exponent_evolved[4:6,4:6]#factor 1/2 to avoid double counting in antisymmetrization
            rho_exponent_evolved_A[:2,2:4] = rho_exponent_evolved[2:4,4:6]
            rho_exponent_evolved_A -= rho_exponent_evolved_A.T

            rho_exponent_evolved_R[:2,:2] = rho_exponent_evolved[:2,2:4]
            rho_exponent_evolved_R[:2,2:4] = rho_exponent_evolved[:2,4:6]
            rho_exponent_evolved_R[2:4,:2] = rho_exponent_evolved[6:8,4:6]
            rho_exponent_evolved_R[2:4,2:4] = rho_exponent_evolved[6:8,4:6]

            rho_exponent_evolved_C[:2,:2] = 0.5*rho_exponent_evolved[:2,:2]
            rho_exponent_evolved_C[2:4,2:4] = 0.5*rho_exponent_evolved[6:8,6:8]
            rho_exponent_evolved_C[:2,2:4] = rho_exponent_evolved[:2,6:8]
            rho_exponent_evolved_C -= rho_exponent_evolved_C.T

            rho_exponent_evolved_A_inv = linalg.inv(rho_exponent_evolved_A)
            rho_exponent_evolved = 0.5*(rho_exponent_evolved_R @ rho_exponent_evolved_A_inv @ rho_exponent_evolved_R.T + rho_exponent_evolved_C )
            

            rho_evolved = np.zeros((4,4),dtype=np.clongdouble)
            # no minus signs because of sign-change-convention for overlap applies only to IM and here we compute DM of system, factor 2 bc of antisymmetry
            a1 =   2 * rho_exponent_evolved[2,3]#
            a2 = +2 * rho_exponent_evolved[2,0]
            a3 = 2 * rho_exponent_evolved[2,1]#
            a4 = 2 * rho_exponent_evolved[3,0]#
            a5 = +2 * rho_exponent_evolved[3,1]
            a6 = 2 * rho_exponent_evolved[0,1]#
            
            rho_evolved[0,0] = 1
            rho_evolved[0,3] = - a5

            rho_evolved[1,1] = a6
            rho_evolved[1,2] = - a4

            rho_evolved[2,1] = a3
            rho_evolved[2,2] = a1

            rho_evolved[3,0] = a2
            rho_evolved[3,3] = a1*a6 - a2*a5 + a3*a4
        
                
            #print(abs(np.sqrt(linalg.det(rho_exponent_evolved_A_inv)) *np.sqrt(linalg.det(A))))
            rho_evolved *=  abs(np.sqrt(linalg.det(rho_exponent_evolved_A )) * np.sqrt(linalg.det(A * np.power(0.5,4.*delt/A.shape[0])))) * norm_IM**2 *  1./(1+np.exp(-beta_down)) *  1./(1+np.exp(-beta_up))#norm squared because of two environments
            #print(np.sqrt(linalg.det(rho_exponent_evolved_A )) , np.sqrt(linalg.det(A *np.power(0.5,4*(delt//2-2) / (A.shape[0]))) ) )#norm squared because of two environments)
            #print(np.amax(A),linalg.det(A * np.power(0.5,2*(delt//2-2) / (A.shape[0]))),linalg.det(rho_exponent_evolved_A ))
            #np.power(0.5,delt//2-2) *
            #*np.sqrt(linalg.det(A * np.power(0.5,2*(delt//2-2) / (A.shape[0])) )))
            #print(np.power(0.5,(delt//2-2) / A.shape[0]))
            trace_vals[intermediate_time_dm]=( np.trace(rho_evolved))
            #rho_eigvals = linalg.eigvals(rho_evolved)
            #rho_eigvals_max[intermediate_time_dm]=(np.max(rho_eigvals))
            #rho_eigvals_min[intermediate_time_dm]=(np.min(rho_eigvals))
            rho_00[intermediate_time_dm]=(np.real(rho_evolved[0,0]))
            rho_11[intermediate_time_dm]=(np.real(rho_evolved[1,1]))
            rho_22[intermediate_time_dm]=(np.real(rho_evolved[2,2]))
            rho_33[intermediate_time_dm]=(np.real(rho_evolved[3,3]))
            rho_full[intermediate_time_dm,:,:]=np.real(rho_evolved[:,:]) #/ trace_vals[intermediate_time_dm])
            times[intermediate_time_dm]=intermediate_time_dm
            #print(intermediate_time_dm, rho_evolved )/ trace_vals[intermediate_time_dm])
        
            with h5py.File(filename+'_spinfulpropag' + ".hdf5", 'a') as f:
                propag_data = f['propag_IM']
            
                propag_data[intermediate_time_dm] = rho_00[intermediate_time_dm]#this is used when comparing to Michael (from DM)
                propag[intermediate_time_dm] = propag_data[intermediate_time_dm]
                data = f['rho_00'] 
                data[intermediate_time_dm] = rho_00[intermediate_time_dm]
                data = f['rho_11'] 
                data[intermediate_time_dm] = rho_11[intermediate_time_dm]
                data = f['rho_22'] 
                data[intermediate_time_dm] = rho_22[intermediate_time_dm]
                data = f['rho_33'] 
                data[intermediate_time_dm] = rho_33[intermediate_time_dm]
                data = f['trace'] 
                data[intermediate_time_dm] = trace_vals[intermediate_time_dm]
                data = f['max_eigvals'] 
                data[intermediate_time_dm] = rho_eigvals_max[intermediate_time_dm]
                data = f['min_eigvals'] 
                data[intermediate_time_dm] = rho_eigvals_min[intermediate_time_dm]
                data = f['rho_full'] 
                data[intermediate_time_dm,:,:] = rho_full[intermediate_time_dm,:,:]

                times_data = f['propag_times']
                times_data[intermediate_time_dm] = times[intermediate_time_dm]
            

    elif mode == 2:#individual final-time IM for each evolution time:

        exponent_check = exponent.copy()
        #impurity
        #hopping between spin species (gate is easily found by taking kernel of xy gate at isotropic parameters):
        seed(10)
        for i in range(dim_B//4-1):
            #t =  0.57*np.cos(0.42 * i)
        
            mu_up =0#0.3*delta_t# 0.3*np.sin(2.2 * i)
            mu_down =0#0.3*delta_t# 0.18*np.sin(1.82 * i)
            #mu_up =0# random()
            #mu_down =0# random()
            #print(t,mu_up,mu_down)

            T=1+np.tan(t/2)**2
            # forward 
            # (matrix elements between up -> down)
            exponent_check[dim_B//2 - 3 - 2*i, 3*dim_B + dim_B//2 - 2 - 2*i] += 1.j * np.tan(t/2) *2/T *np.exp(1.j * mu_up)
            exponent_check[dim_B//2 - 2 - 2*i, 3*dim_B + dim_B//2 - 3 - 2*i] -= 1.j * np.tan(t/2)*2/T *np.exp(1.j * mu_down)
            #(matrix elements between up -> up)
            exponent_check[dim_B//2 - 3 - 2*i, dim_B//2 - 2 - 2*i] += 1 *np.cos(t) *np.exp(1.j * mu_up)
            exponent_check[3*dim_B + dim_B//2 - 3 - 2*i, 3*dim_B + dim_B//2 - 2 - 2*i] += 1 *np.cos(t) *np.exp(1.j * mu_down)

            # forward Transpose (antisymm)
            exponent_check[3*dim_B + dim_B//2 - 2 - 2*i, dim_B//2 - 3 - 2*i] += -1.j * np.tan(t/2)*2/T *np.exp(1.j * mu_up)
            exponent_check[3*dim_B + dim_B//2 - 3 - 2*i, dim_B//2 - 2 - 2*i] -= -1.j * np.tan(t/2)*2/T *np.exp(1.j * mu_down)
            exponent_check[dim_B//2 - 2 - 2*i,dim_B//2 - 3 - 2*i] += -1 *np.cos(t) *np.exp(1.j * mu_up)
            exponent_check[3*dim_B + dim_B//2 - 2 - 2*i, 3*dim_B + dim_B//2 - 3 - 2*i] += -1 *np.cos(t) *np.exp(1.j * mu_down)

            # backward
            exponent_check[dim_B//2 + 1 + 2*i, 3*dim_B + dim_B//2 + 2 + 2*i] += - 1.j * np.tan(t/2)*2/T *np.exp(-1.j * mu_down)
            exponent_check[dim_B//2 + 2 + 2*i, 3*dim_B + dim_B//2 + 1 + 2*i] -= - 1.j * np.tan(t/2)*2/T *np.exp(-1.j * mu_up)
            exponent_check[dim_B//2 + 1 + 2*i, dim_B//2 + 2 + 2*i] += 1 *np.cos(t) * np.exp(-1.j * mu_up)
            exponent_check[3*dim_B + dim_B//2 + 1 + 2*i, 3*dim_B + dim_B//2 + 2 + 2*i] += 1 *np.cos(t) * np.exp(-1.j * mu_down)

            # backward Transpose (antisymm)
            exponent_check[3*dim_B + dim_B//2 + 2 + 2*i, dim_B//2 + 1 + 2*i] += + 1.j * np.tan(t/2)*2/T *np.exp(-1.j * mu_down)
            exponent_check[3*dim_B + dim_B//2 + 1 + 2*i, dim_B//2 + 2 + 2*i] -= + 1.j * np.tan(t/2)*2/T *np.exp(-1.j * mu_up)
            exponent_check[dim_B//2 + 2 + 2*i, dim_B//2 + 1 + 2*i] += -1 *np.cos(t) * np.exp(-1.j * mu_up)
            exponent_check[3*dim_B + dim_B//2 + 2 + 2*i, 3*dim_B + dim_B//2 + 1 + 2*i] += -1 *np.cos(t) * np.exp(-1.j * mu_down)
        
        
        #for spin up/down density matrix
        A = np.bmat([[exponent_check[1:dim_B-1,1:dim_B-1], exponent_check[1:dim_B-1,dim_B+1 :2*dim_B-1], exponent_check[1:dim_B-1,2*dim_B +1:3*dim_B-1], exponent_check[1:dim_B-1,3*dim_B+1:4*dim_B-1]], 
                    [exponent_check[dim_B+1 :2*dim_B-1,1:dim_B-1], exponent_check[dim_B+1 :2*dim_B-1,dim_B+1 :2*dim_B-1], exponent_check[dim_B+1 :2*dim_B-1,2*dim_B +1:3*dim_B-1], exponent_check[dim_B+1 :2*dim_B-1,3*dim_B+1:4*dim_B-1]],
                    [exponent_check[2*dim_B +1:3*dim_B-1,1:dim_B-1], exponent_check[2*dim_B +1:3*dim_B-1,dim_B+1 :2*dim_B-1], exponent_check[2*dim_B +1:3*dim_B-1,2*dim_B +1:3*dim_B-1], exponent_check[2*dim_B +1:3*dim_B-1,3*dim_B+1:4*dim_B-1]],
                    [exponent_check[3*dim_B+1:4*dim_B-1,1:dim_B-1], exponent_check[3*dim_B+1:4*dim_B-1,dim_B+1 :2*dim_B-1], exponent_check[3*dim_B+1:4*dim_B-1,2*dim_B +1:3*dim_B-1], exponent_check[3*dim_B+1:4*dim_B-1,3*dim_B+1:4*dim_B-1]]])

    
        
        R = np.bmat([exponent_check[[0,dim_B -1,dim_B,2*dim_B -1, 2*dim_B, 3*dim_B -1,3*dim_B, 4*dim_B -1],1:dim_B-1] ,exponent_check[[0,dim_B -1,dim_B,2*dim_B -1, 2*dim_B, 3*dim_B -1,3*dim_B, 4*dim_B -1],dim_B+1:2*dim_B-1],exponent_check[[0,dim_B -1,dim_B,2*dim_B -1, 2*dim_B, 3*dim_B -1,3*dim_B, 4*dim_B -1],2*dim_B+1:3*dim_B-1],exponent_check[[0,dim_B -1,dim_B,2*dim_B -1, 2*dim_B, 3*dim_B -1,3*dim_B, 4*dim_B -1],3*dim_B+1:4*dim_B-1] ])
        C = np.zeros((8,8),dtype=np.clongdouble)
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

        rho_evolved = np.zeros((4,4),dtype=np.clongdouble)

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

        rho_evolved *= 1./((1+np.exp(-beta_up))) *1./((1+np.exp(-beta_down))) * np.sqrt(linalg.det(A)) * norm_IM**2#norm squared because of two environments

        trace_vals[iter]=( np.trace(rho_evolved))
        rho_eigvals = linalg.eigvals(rho_evolved)
        rho_eigvals_max[iter]=(np.max(rho_eigvals))
        rho_eigvals_min[iter]=(np.min(rho_eigvals))
        rho_00[iter]=(np.real(rho_evolved[0,0]))
        rho_11[iter]=(np.real(rho_evolved[1,1]))
        rho_22[iter]=(np.real(rho_evolved[2,2]))
        rho_33[iter]=(np.real(rho_evolved[3,3]))
        rho_full[iter,:,:]=np.real(rho_evolved[:,:])
        times[iter]= iter + 1
        

        with h5py.File(filename+'_spinfulpropag' + ".hdf5", 'a') as f:
                propag_data = f['propag_IM']
            
                propag_data[iter] = rho_00[iter]#this is used when comparing to Michael (from DM)
                propag[iter] = propag_data[iter]
                data = f['rho_00'] 
                data[iter] = rho_00[iter]
                data = f['rho_11'] 
                data[iter] = rho_11[iter]
                data = f['rho_22'] 
                data[iter] = rho_22[iter]
                data = f['rho_33'] 
                data[iter] = rho_33[iter]
                data = f['trace'] 
                data[iter] = trace_vals[iter]
                data = f['max_eigvals'] 
                data[iter] = rho_eigvals_max[iter]
                data = f['min_eigvals'] 
                data[iter] = rho_eigvals_min[iter]
                data = f['rho_full'] 
                data[iter,:,:] = rho_full[iter,:,:]

                times_data = f['propag_times']
                times_data[iter] = times[iter]


    elif mode == 3:#compute propagator, not density matrix
        exponent_check = exponent.copy()
        #impurity
        #hopping between spin species (gate is easily found by taking kernel of xy gate at isotropic parameters):
        seed(10)
        for i in range(dim_B//4-1):
            #t =  0.57*np.cos(0.42 * i)
        
            mu_up =0#4*delta_t#0.3*delta_t# 0.3*np.sin(2.2 * i)
            mu_down =0#4*delta_t#0.3*delta_t# 0.18*np.sin(1.82 * i)
            #mu_up =0# random()
            #mu_down =0# random()
            #print(t,mu_up,mu_down)

            T=1+np.tan(t/2)**2
            # forward 
            # (matrix elements between up -> down)
            exponent_check[dim_B//2 - 3 - 2*i, 3*dim_B + dim_B//2 - 2 - 2*i] += 1.j * np.tan(t/2) *2/T *np.exp(1.j * mu_up)
            exponent_check[dim_B//2 - 2 - 2*i, 3*dim_B + dim_B//2 - 3 - 2*i] -= 1.j * np.tan(t/2)*2/T *np.exp(1.j * mu_down)
            #(matrix elements between up -> up)
            exponent_check[dim_B//2 - 3 - 2*i, dim_B//2 - 2 - 2*i] += 1 *np.cos(t) *np.exp(1.j * mu_up)
            exponent_check[3*dim_B + dim_B//2 - 3 - 2*i, 3*dim_B + dim_B//2 - 2 - 2*i] += 1 *np.cos(t) *np.exp(1.j * mu_down)

            # forward Transpose (antisymm)
            exponent_check[3*dim_B + dim_B//2 - 2 - 2*i, dim_B//2 - 3 - 2*i] += -1.j * np.tan(t/2)*2/T *np.exp(1.j * mu_up)
            exponent_check[3*dim_B + dim_B//2 - 3 - 2*i, dim_B//2 - 2 - 2*i] -= -1.j * np.tan(t/2)*2/T *np.exp(1.j * mu_down)
            exponent_check[dim_B//2 - 2 - 2*i,dim_B//2 - 3 - 2*i] += -1 *np.cos(t) *np.exp(1.j * mu_up)
            exponent_check[3*dim_B + dim_B//2 - 2 - 2*i, 3*dim_B + dim_B//2 - 3 - 2*i] += -1 *np.cos(t) *np.exp(1.j * mu_down)

            # backward
            exponent_check[dim_B//2 + 1 + 2*i, 3*dim_B + dim_B//2 + 2 + 2*i] += - 1.j * np.tan(t/2)*2/T *np.exp(-1.j * mu_down)
            exponent_check[dim_B//2 + 2 + 2*i, 3*dim_B + dim_B//2 + 1 + 2*i] -= - 1.j * np.tan(t/2)*2/T *np.exp(-1.j * mu_up)
            exponent_check[dim_B//2 + 1 + 2*i, dim_B//2 + 2 + 2*i] += 1 *np.cos(t) * np.exp(-1.j * mu_up)
            exponent_check[3*dim_B + dim_B//2 + 1 + 2*i, 3*dim_B + dim_B//2 + 2 + 2*i] += 1 *np.cos(t) * np.exp(-1.j * mu_down)

            # backward Transpose (antisymm)
            exponent_check[3*dim_B + dim_B//2 + 2 + 2*i, dim_B//2 + 1 + 2*i] += + 1.j * np.tan(t/2)*2/T *np.exp(-1.j * mu_down)
            exponent_check[3*dim_B + dim_B//2 + 1 + 2*i, dim_B//2 + 2 + 2*i] -= + 1.j * np.tan(t/2)*2/T *np.exp(-1.j * mu_up)
            exponent_check[dim_B//2 + 2 + 2*i, dim_B//2 + 1 + 2*i] += -1 *np.cos(t) * np.exp(-1.j * mu_up)
            exponent_check[3*dim_B + dim_B//2 + 2 + 2*i, 3*dim_B + dim_B//2 + 1 + 2*i] += -1 *np.cos(t) * np.exp(-1.j * mu_down)
        
        
        exponent_inv = linalg.inv(exponent_check)


        with h5py.File(filename+'_spinfulpropag' + ".hdf5", 'a') as f:
            propag_data = f['propag_IM']
            tau0=0
            for tau in range (tau0,nbr_Floquet_layers):
                delt = 2 * (total_time - tau)
                print(tau,exponent_inv[2*dim_B + delt+1, delt+1] )
                #propag_data[iter] = exponent_inv[2*dim_B + 1,dim_B//2-1]#this is the propagator for the spin up (use this when comparing to analytical calculation from scratch!)
                #propag_data[tau - tau0] = exponent_inv[ 2 * dim_B + dim_B//2 -2*tau -1, dim_B//2 -2*tau0 -1] - exponent_inv[dim_B//2 -2*tau0 -1, 2 * dim_B + dim_B//2 -2*tau -1]
                propag_data[tau] = exponent_inv[2*dim_B + delt+1, delt+1]  #<n(t)># works, used for benchmark
                #propag_data[iter] =  exponent_inv[2*dim_B + dim_B//2 -2*iter -2 , dim_B//2 -2*iter -2]  -  exponent_inv[2*dim_B + dim_B//2 -2*iter -1 , dim_B//2 -2*iter -1]#current 
                #propag_data[iter] = -exponent_inv[2*dim_B + dim_B//2 -2*iter -1 , dim_B//2 -2*iter -1]# this is what I take to compare with cont. time
                times_data = f['propag_times']
                times_data[tau - tau0] = tau + 1 -tau0 #nbr_Floquet_layers




plt.plot(np.arange(2,200)*delta_t* Gamma, trace_vals[2:200],linewidth=2,label='Tr'+r'$(\rho)$')
plt.plot(np.arange(2,200)*delta_t* Gamma, rho_eigvals_max[2:200],linewidth=2,linestyle= '-',label='max. eigenvalue of '+ r'$\rho$')
plt.plot(np.arange(2,200)*delta_t* Gamma, rho_eigvals_min[2:200],linewidth=2,label='min. eigenvalue of '+ r'$\rho$')
plt.plot(np.arange(2,200)*delta_t* Gamma, rho_00[2:200],linewidth=2,linestyle= '--',label=r'$\rho_{00}$')
plt.plot(np.arange(2,200)*delta_t * Gamma, rho_11[2:200],linewidth=2,linestyle= 'dotted',label=r'$\rho_{11}$')
plt.plot(np.arange(2,200)*delta_t* Gamma, rho_22[2:200],linewidth=2,linestyle= '--',alpha=0.8,label=r'$\rho_{22}$')
#plt.plot(times[1::interval]*delta_t* Gamma,propag[1::interval],linewidth=2,linestyle= '--',alpha=0.8,label='Our IM result via Grassmanns')
plt.plot(np.arange(2,200)*delta_t* Gamma, rho_33[2:200],linewidth=2,linestyle= 'dotted',label=r'$\rho_{33}$')
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