import scipy.integrate as integrate
import scipy.special as special
from tests import anti_sym_check
import numpy as np
from numpy import linalg
from create_correlation_block import create_correlation_block
import h5py

np.set_printoptions(suppress=False, linewidth=np.nan)
np.set_printoptions(linewidth=np.nan, precision=6, suppress=True)

#create B in Michael's basis

min_time=50
max_time=201
interval =50
conv = 'M'
time_scheme = ''#'ct'#'ct' is only needed for a single environment. Otherwise , the ct-convention is adopted be the join-method.
int_lim_low = -20
int_lim_up = 20

filename_comp = '/Users/julianthoenniss/Documents/PhD/data/compmode=C_o=1_Jx=1.0_Jy=1.0_g=0.0mu=0.0_del_t=0.01_beta=0.0_L=100_init=2'
#filename_comp = '/Users/julianthoenniss/Documents/PhD/data/Jx=0.1_Jy=0.1_g=0.0mu=0.0_del_t=1.0_L=200_FermiSea'
filename = '/Users/julianthoenniss/Documents/PhD/data/Millis_mu=-.2_timestep=0.1_T=50-200_hyb=0.05_D=10'

if conv == 'J':
    filename += '_my_conv' 
    print('using Js convention')
elif conv == 'M':
    filename += '_Michaels_conv' 
    print('using Ms convention')

if time_scheme == 'ct':
    filename += '_continuoustime' 
    print('using continuous time convention')


with h5py.File(filename + ".hdf5", 'w') as f:
        dset_IM_exponent = f.create_dataset('IM_exponent', ((max_time - min_time)//interval + 1, 4 * (max_time-1), 4 * (max_time-1)),dtype=np.complex_)
        dset_times = f.create_dataset('times', ((max_time - min_time)//interval + 1,),dtype=np.int16)


global_gamma = 1.
delta_t = 0.1
mu=-0.2 * global_gamma
beta = 10/0.05
def spec_dens(gamma,energy):
    e_c = 1*gamma 
    nu = 10/gamma
    #return 0.05  * gamma /((1+np.exp(nu*(energy - e_c))) * (1+np.exp(-nu*(energy + e_c)))) 
    return 0.05 


def g_a_real(energy,beta,mu,t,t_prime):
    return np.real(1./(1+np.exp(beta * (energy - mu))) * np.exp(-1.j* energy *(t-t_prime)))
def g_a_imag(energy,beta,mu,t,t_prime):
    return np.imag(1./(1+np.exp(beta * (energy - mu))) * np.exp(-1.j* energy *(t-t_prime)))

def g_b_real(energy,beta,mu,t,t_prime):
    return np.real((1./(1+np.exp(beta * (energy - mu))) - 1) * np.exp(-1.j* energy *(t-t_prime)))
def g_b_imag(energy,beta,mu,t,t_prime):
    return np.imag((1./(1+np.exp(beta * (energy - mu))) - 1) * np.exp(-1.j* energy *(t-t_prime)))

#print(integrate.quad(lambda x:  spec_dens(global_gamma,x) , int_lim_low, int_lim_up))

#print(integrate.quad(lambda x: g_a(x,1./global_gamma,mu,0,0), int_lim_low, int_lim_up))
#print(integrate.quad(lambda x: g_b(x,1./global_gamma,mu,0,0), int_lim_low, int_lim_up))
#print(integrate.quad(lambda x: 1/(2*np.pi) * spec_dens(global_gamma,x) * g_a(x,1./global_gamma,mu,0,0), int_lim_low, int_lim_up))
iter = 0
for nbr_Floquet_layers in range (min_time,max_time,interval):

    """
    B_comp = np.zeros((4*(nbr_Floquet_layers),4*(nbr_Floquet_layers)), dtype=np.complex_)
    with h5py.File(filename_comp + '.hdf5', 'r') as f:
        B_comp = f['IM_exponent'][nbr_Floquet_layers-1,:4*(nbr_Floquet_layers),:4*(nbr_Floquet_layers)]

    S = np.zeros(B_comp.shape,dtype=np.complex_)
    for i in range (nbr_Floquet_layers):#order plus and minus next to each other
        S [B_comp.shape[0] // 2 - (2 * i) - 2,4 * i] = 1
        S [B_comp.shape[0] // 2 - (2 * i) - 1,4 * i + 2] = 1
        S [B_comp.shape[0] // 2 + (2 * i) ,4 * i + 1] = 1
        S [B_comp.shape[0] // 2 + (2 * i) + 1,4 * i + 3] = 1
    
    B_comp = S @ B_comp @ S.T

    #the following two transformation bring it into in/out- basis (not theta, zeta)
    rot = np.zeros(B_comp.shape)
    for i in range(0,B_comp.shape[0], 2):#go from bar, nonbar to zeta, theta
        rot[i,i] = 1./np.sqrt(2)
        rot[i,i+1] = 1./np.sqrt(2)
        rot[i+1,i] = - 1./np.sqrt(2) * np.sign(2*nbr_Floquet_layers - i-1)
        rot[i+1,i+1] = 1./np.sqrt(2) * np.sign(2*nbr_Floquet_layers - i-1)
    B_comp = rot.T @ B_comp @ rot

    U = np.zeros(B_comp.shape)#order in the way specified in pdf for him (forward,backward,forward,backward,...)
    for i in range (nbr_Floquet_layers):
        U[4*i, B_comp.shape[0] //2 - (2*i) -1] = 1
        U[4*i + 1, B_comp.shape[0] //2 + (2*i)] = 1
        U[4*i + 2, B_comp.shape[0] //2 - (2*i) -2] = 1
        U[4*i + 3, B_comp.shape[0] //2 + (2*i) + 1] = 1
    B_comp = U @ B_comp @ U.T 
    """    
    print('computing B..')
    B = np.zeros((4*nbr_Floquet_layers,4*nbr_Floquet_layers),dtype=np.complex_)

    for j in range (nbr_Floquet_layers):
        for i in range (j+1,nbr_Floquet_layers):
            
            t = i * delta_t
            t_prime = j * delta_t
            f_a_real = integrate.quad(lambda x: 1./(2*np.pi) * spec_dens(global_gamma,x) * g_a_real(x,beta,mu,t,t_prime), int_lim_low, int_lim_up)
            f_a_imag = integrate.quad(lambda x: 1./(2*np.pi) * spec_dens(global_gamma,x) * g_a_imag(x,beta,mu,t,t_prime), int_lim_low, int_lim_up)
            integ_a = f_a_real[0] + 1.j*f_a_imag[0]
            f_b_real = integrate.quad(lambda x: 1./(2*np.pi) * spec_dens(global_gamma,x) * g_b_real(x,beta,mu,t,t_prime), int_lim_low, int_lim_up)
            f_b_imag = integrate.quad(lambda x: 1./(2*np.pi) * spec_dens(global_gamma,x) * g_b_imag(x,beta,mu,t,t_prime), int_lim_low, int_lim_up)
            integ_b = f_b_real[0] + 1.j*f_b_imag[0]
            
            B[4*i,4*j+1] = - np.conj(integ_b) * delta_t**2
            B[4*i,4*j+2] = - np.conj(integ_a) * delta_t**2
            B[4*i+1,4*j] = integ_b * delta_t**2
            B[4*i+1,4*j+3] = integ_a * delta_t**2
            B[4*i+2,4*j] =  integ_b * delta_t**2
            B[4*i+2,4*j+3] = integ_a * delta_t**2
            B[4*i+3,4*j+1] = - np.conj(integ_b) * delta_t**2
            B[4*i+3,4*j+2] = - np.conj(integ_a) * delta_t**2
            

        #for equal time
        f_a_real = integrate.quad(lambda x: 1./(2*np.pi) * spec_dens(global_gamma,x) * g_a_real(x,beta,mu,0,0), int_lim_low, int_lim_up)
        f_a_imag = integrate.quad(lambda x: 1./(2*np.pi) * spec_dens(global_gamma,x) * g_a_imag(x,beta,mu,0,0), int_lim_low, int_lim_up)
        integ_a = f_a_real[0] + 1.j*f_a_imag[0]
        
        f_b_real = integrate.quad(lambda x: 1./(2*np.pi) * spec_dens(global_gamma,x) * g_b_real(x,beta,mu,0,0), int_lim_low, int_lim_up)
        f_b_imag = integrate.quad(lambda x: 1./(2*np.pi) * spec_dens(global_gamma,x) * g_b_imag(x,beta,mu,0,0), int_lim_low, int_lim_up)
        integ_b = f_b_real[0] + 1.j*f_b_imag[0]

        
        B[4*j+1,4*j] =  integ_b * delta_t**2
        B[4*j+2,4*j] =  integ_b * delta_t**2
        B[4*j+3,4*j+1] = - np.conj(integ_b) * delta_t**2
        B[4*j+3,4*j+2] = - np.conj(integ_a) * delta_t**2
        
        
        # the plus and minus one here come from the overlap of GMs
        B[4*j+2,4*j] += 1 
        B[4*j+3,4*j+1] -=1 

    B += - B.T#like this, one obtains 2*exponent, as in other code.
    #choice of basis: Michael's convention
    #B=B_comp
    anti_sym_check(B)



    #for continuous time
    if time_scheme == 'ct':
        dim_B = B.shape[0]

        U = np.zeros(B.shape)#order in the way specified in pdf for him (forward,backward,forward,backward,...)
        for i in range (B.shape[0] //4):
            U[4*i, B.shape[0] //2 - (2*i) -1] = 1
            U[4*i + 1, B.shape[0] //2 + (2*i)] = 1
            U[4*i + 2, B.shape[0] //2 - (2*i) -2] = 1
            U[4*i + 3, B.shape[0] //2 + (2*i) + 1] = 1
        B = U.T @ B @ U
        
        
        print('Rotated from M basis to Grassmann basis')

        #here, the matrix B is in the Grassmann convention
        #subtract ones from overlap, these will be included in the impurity MPS, instead
        for i in range (dim_B//2):
            B[2*i, 2*i+1] -= 1 
            B[2*i+1, 2*i] += 1 
        
        dim_B += 4#update for embedding into larger matrix
        B_enlarged = np.zeros((dim_B,dim_B),dtype=np.complex_)
        B_enlarged[1:dim_B//2-1, 1:dim_B//2-1] = 0.5 * B[:B.shape[0]//2,:B.shape[0]//2]#0.5 to avoid double couting in antisymmetrization
        B_enlarged[dim_B//2+1:dim_B-1, 1:dim_B//2-1] = B[B.shape[0]//2:, :B.shape[0]//2]
        B_enlarged[dim_B//2+1:dim_B-1, dim_B//2+1:dim_B-1] = 0.5 * B[B.shape[0]//2:, B.shape[0]//2:]
        
        #include ones from grassmann identity-resolutions (not included in conventional IM)
        for i in range (0,dim_B//2):
            B_enlarged[2*i, 2*i+1] += 1 

        B_enlarged += - B_enlarged.T

        B_enlarged_inv = linalg.inv(B_enlarged)

        B = B_enlarged

        # adjust signs that make the influence matrix a vectorized state (in Grassmann basis)
        for i in range (dim_B):
            for j in range (dim_B):
                if (i+j)%2 == 1:
                    B[i,j] *= -1

        U = np.zeros(B.shape)#order in the way specified in pdf for him (forward,backward,forward,backward,...)
        for i in range (B.shape[0] //4):
            U[4*i, B.shape[0] //2 - (2*i) -1] = 1
            U[4*i + 1, B.shape[0] //2 + (2*i)] = 1
            U[4*i + 2, B.shape[0] //2 - (2*i) -2] = 1
            U[4*i + 3, B.shape[0] //2 + (2*i) + 1] = 1
        B = U @ B @ U.T
        print('Rotated from Grassmann basis to Ms basis')

    nbr_Floquet_layers_effective = nbr_Floquet_layers

    if time_scheme == 'ct':
        nbr_Floquet_layers_effective += 1

    with h5py.File(filename + '.hdf5', 'a') as f:
        IM_data = f['IM_exponent']
        IM_data[iter ,:B.shape[0],:B.shape[0]] = B[:,:]
        times_data = f['times']
        times_data[iter] = nbr_Floquet_layers_effective

    correlation_block = create_correlation_block(B, nbr_Floquet_layers_effective, filename)

    print(np.real(linalg.eigvals(correlation_block)))

    iter += 1
#print(B)