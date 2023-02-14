import scipy.integrate as integrate
import scipy.special as special
from tests import anti_sym_check
import numpy as np
from numpy import linalg
from create_correlation_block import create_correlation_block
import h5py
from real_time_IM_funcs import spec_dens, g_a_real, g_a_imag, g_b_real, g_b_imag, create_correlation_block

np.set_printoptions(suppress=False, linewidth=np.nan)
np.set_printoptions(linewidth=np.nan, precision=6, suppress=True)


#########################Define parameters###################################

#___________________time steps______________________________
delta_t = 0.05 #Physical evolution time for one time step in environment
T_ren = 1#number of "sub"-timesteps into which a physical timestep in the environment is divided
delta_t *= 1./T_ren#this is the effective time step needed for the computation

min_time=6#minimal time for which IF should be computed
max_time=7#maximal time for which IF should be computed
interval =1#interval with which time steps should be computed

#__________________physical parameters______________________________
global_gamma = 1.#global energy scale
mu=0#for Cohen 2015, set this to 0
beta = 50./global_gamma #for Cohen 2015, set this to 50./global_gamma

#integration limit (should be adjusted depending on the band width)
int_lim_low = -12#for Cohen 2015, set this to -12
int_lim_up = 12#for Cohen 2015, set this to 12

#_____________________other___________________________________
#contraction scheme:
#for the "simultaneous-evolution-scheme" (as opposed to successive env.-impurity evolution). For the "continuous-time" IF (defined for successive contraseion), leave this empty
time_scheme = ''#'se' is only needed for a single environment. Otherwise , the se-convention is adopted be the join-method.
#output file:
filename = '/Users/julianthoenniss/Documents/PhD/data/Cohen2015_inchworm_deltat=0.005_shorttime_doublhyb_t=2.0'
if time_scheme == 'se':
    filename += '_sim' 
    print('using simultaneous-contraction convention')
##################################################################################


#create file in which the IF is stored
with h5py.File(filename + ".hdf5", 'w') as f:
        dset_IM_exponent = f.create_dataset('IM_exponent', ((max_time - min_time)//interval + 1, 4 * (max_time-1), 4 * (max_time-1)),dtype=np.complex_)
        dset_times = f.create_dataset('times', ((max_time - min_time)//interval + 1,),dtype=np.int16)


iter = 0
for nbr_Floquet_layers in range (min_time,max_time,interval):#loop that individually creates the IF for a given number of Floquet layers
    dim_B = 4*nbr_Floquet_layers# Size of exponent matrix in the IF. The variable dim_B is equal to 4*nbr_Floquet_layers
    dim_B_temp = dim_B * T_ren#effective size of exponent matrix in the IF if physical time steps are each subdivided into T_ren substeps
    
    print('computing B..')
    B = np.zeros((dim_B_temp,dim_B_temp),dtype=np.complex_)

    for j in range (dim_B_temp//4):
        for i in range (j+1,dim_B_temp//4):
            
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

    B += - B.T#like this, one obtains 2*exponent, needed for Grassmann code.
    #here, the IF is in the folded basis: (in-fw, in-bw, out-fw, out-bw,...) which is needed for correlation matrix

    if T_ren > 0: #if physical time steps are subdivided, integrate out the "auxiliary" legs
        
        #rotate from folded basis into Grassmann basis
        U = np.zeros((dim_B_temp,dim_B_temp))
        for i in range (dim_B_temp //4):
            U[4*i, dim_B_temp //2 - (2*i) -1] = 1
            U[4*i + 1, dim_B_temp //2 + (2*i)] = 1
            U[4*i + 2, dim_B_temp //2 - (2*i) -2] = 1
            U[4*i + 3, dim_B_temp //2 + (2*i) + 1] = 1
        B = U.T @ B @ U
        
        #add intermediate integration measure to integrate out internal legs
        for i in range (dim_B//2):
            for j in range (T_ren-1):
                B[2*i*T_ren + 1 + 2*j,2*i*T_ren+2+ 2*j] += 1  
                B[2*i*T_ren+2+ 2*j,2*i*T_ren + 1 + 2*j] += -1  

        #select submatrix that contains all intermediate times that are integrated out
        B_sub =  np.zeros((dim_B_temp - dim_B, dim_B_temp - dim_B),dtype=np.complex_)
        for i in range (dim_B//2 ):
            for j in range (dim_B//2 ):
                B_sub[i*(2*T_ren-2):i*(2*T_ren-2 )+2*T_ren-2,j*(2*T_ren-2):j*(2*T_ren-2 )+2*T_ren-2] = B[2*i*T_ren+1:2*(i*T_ren + T_ren)-1,2*j*T_ren+1:2*(j*T_ren + T_ren)-1]

        #matrix coupling external legs to integrated (internal) legs
        B_coupl =  np.zeros((dim_B_temp - dim_B,dim_B),dtype=np.complex_)
        for i in range (dim_B//2 ):
            for j in range (dim_B//2 ):
                B_coupl[i*(2*T_ren-2):i*(2*T_ren-2 )+2*T_ren-2,2*j] = B[2*i*T_ren+1:2*(i*T_ren + T_ren)-1,2*j*T_ren]
                B_coupl[i*(2*T_ren-2):i*(2*T_ren-2 )+2*T_ren-2,2*j+1] = B[2*i*T_ren+1:2*(i*T_ren + T_ren)-1,2*(j+1)*T_ren-1]

        #part of matriy that is neither integrated nor coupled to integrated variables
        B_ext = np.zeros((dim_B,dim_B),dtype=np.complex_)
        for i in range (dim_B//2 ):
            for j in range (dim_B//2 ):
                B_ext[2*i,2*j] = B[2*i*T_ren,2*j*T_ren]
                B_ext[2*i+1,2*j] = B[2*(i+1)*T_ren-1,2*j*T_ren]
                B_ext[2*i,2*j+1] = B[2*i*T_ren,2*(j+1)*T_ren-1]
                B_ext[2*i+1,2*j+1] = B[2*(i+1)*T_ren-1,2*(j+1)*T_ren-1]

        B = B_ext + B_coupl.T @ linalg.inv(B_sub) @ B_coupl
        U = np.zeros(B.shape)#order in the way specified in pdf for him (forward,backward,forward,backward,...)
        for i in range (B.shape[0] //4):
            U[4*i, B.shape[0] //2 - (2*i) -1] = 1
            U[4*i + 1, B.shape[0] //2 + (2*i)] = 1
            U[4*i + 2, B.shape[0] //2 - (2*i) -2] = 1
            U[4*i + 3, B.shape[0] //2 + (2*i) + 1] = 1
        B = U @ B @ U.T#rotate from Grassmann basis to folded basis: (in-fw, in-bw, out-fw, out-bw,...)

    #for simultan. contraction scheme
    if time_scheme == 'se':
        dim_B = B.shape[0]

        #rotate from folded basis into Grassmann basis
        U = np.zeros(B.shape)
        for i in range (B.shape[0] //4):
            U[4*i, B.shape[0] //2 - (2*i) -1] = 1
            U[4*i + 1, B.shape[0] //2 + (2*i)] = 1
            U[4*i + 2, B.shape[0] //2 - (2*i) -2] = 1
            U[4*i + 3, B.shape[0] //2 + (2*i) + 1] = 1
        B = U.T @ B @ U
    
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

        #rotate from Grassmann basis to folded basis: (in-fw, in-bw, out-fw, out-bw,...)
        U = np.zeros(B.shape)
        for i in range (B.shape[0] //4):
            U[4*i, B.shape[0] //2 - (2*i) -1] = 1
            U[4*i + 1, B.shape[0] //2 + (2*i)] = 1
            U[4*i + 2, B.shape[0] //2 - (2*i) -2] = 1
            U[4*i + 3, B.shape[0] //2 + (2*i) + 1] = 1
        B = U @ B @ U.T

    nbr_Floquet_layers_effective = nbr_Floquet_layers

    if time_scheme == 'se':
        nbr_Floquet_layers_effective += 1

    with h5py.File(filename + '.hdf5', 'a') as f:
        IM_data = f['IM_exponent']#store in folded basis (in-fw, in-bw, out-fw, out-bw,...)
        IM_data[iter ,:B.shape[0],:B.shape[0]] = B[:,:]
        times_data = f['times']
        times_data[iter] = nbr_Floquet_layers_effective


    #create correlation matrix and store it in a separate file
    correlation_block = create_correlation_block(B, nbr_Floquet_layers_effective, filename)
    #store correlation matrix
    with h5py.File(filename+ "_correlations.hdf5", 'w') as f:
        dset_corr = f.create_dataset('corr_t='+ str(nbr_Floquet_layers_effective), (correlation_block.shape[0],correlation_block.shape[1]),dtype=np.complex_)
        dset_corr[:,:] = correlation_block[:,:]
    print('Correlations stored in block form.')

    iter += 1
