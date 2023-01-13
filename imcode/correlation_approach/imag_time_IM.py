import numpy as np
from scipy import linalg
import h5py
import pandas as pd
import scipy.integrate as integrate
from pfapack import pfaffian as pf
from imag_time_IM_funcs import g_lesser, g_greater, spec_dens, create_correlation_matrix


#___________Set parameters_______________________________________________________________________________________
global_gamma = 1.#global energyscale -> set to 1 
beta = 4./global_gamma# here, twice the value than Benedikt (?)
Gamma = 1.

#spin hopping parameter (set to 0 for Anderson impurity model)
t = 0
#local (spin-dependent) onsite_energies
mu_up =0.
mu_down =0.

dim_B = 500# Size of exponent matrix in the IF. The variable dim_B is equal to 2*nbr_Floquet_layers
delta_t = beta / (dim_B/2)#timestep

#for spec-dens matrix (written in convention of the "guide")
int_lim_low = -1.#integration limits for spectral density
int_lim_up = 1.

#trotter convention:
trotter_convention = 'a' #choose between 'a' (for successive evolution of bath and impurity, see DMFT-guide) or 'b' for the simultaneous impurity-bath evolution

alpha = 0.5 #set this to a values in the interval [0,1]
if trotter_convention == 'b':#for simultaneous evolution, set alpha = 1
    alpha = 1.

###################COMPUTE INFLUENCE FUNCTIONAL AND ITS CORRELATION MATRIX########################################

#________________Compute the matrix B_spec_dens which defines the Influence functional (this corresponds to the equations in this DMFT guide)_____________________________________________________
B_spec_dens = np.zeros((dim_B,dim_B),dtype=np.complex_)
for m in range (B_spec_dens.shape[0]//2):
    tau = m * delta_t
    for n in range (m+1,B_spec_dens.shape[0]//2):
        tau_p = n * delta_t
        B_spec_dens[2*m, 2*n + 1] += - (-1.) * delta_t**2 * integrate.quad(lambda x: 1./(2*np.pi) * spec_dens(global_gamma,x) * g_greater(x,beta,tau_p,tau+delta_t* alpha), int_lim_low, int_lim_up)[0]
        B_spec_dens[2*m+1, 2*n ] += - delta_t**2 * integrate.quad(lambda x: 1./(2*np.pi) * spec_dens(global_gamma,x) * g_lesser(x,beta,tau,tau_p+delta_t* alpha), int_lim_low, int_lim_up)[0]
    B_spec_dens[2*m, 2*m + 1] += - (-1.) * delta_t**2 * integrate.quad(lambda x: 1./(2*np.pi) * spec_dens(global_gamma,x) * g_lesser(x,beta,tau,tau+delta_t* alpha), int_lim_low, int_lim_up)[0]
    B_spec_dens[2*m, 2*m + 1] += - 1. #constant from overlap at m=n
    
B_spec_dens -= B_spec_dens.T#antisymmetrize. This is twice the exponent matrix (this is assumed for later part in the code)

if trotter_convention == 'b':#for simultaneous impurity-bath evolution. Note that the input here is the matrix B_specdens, computed with successive-application-trotter-scheme  
    #remove overlap, it will be included in the impurity gates
    for m in range(B_spec_dens.shape[0]//2):
        B_spec_dens[2*m, 2*m + 1] -= - 1. #constant from overlap at m=n
        B_spec_dens[2*m+1, 2*m] -= + 1. #constant from overlap at m=n

    #account for changes in signs which are introduced in order to keep the impurity gates unchanged for both cases, a and b: change sign of all entries that have one conjugate variable 
    for i in range (B_spec_dens.shape[0]):
        B_spec_dens[i,(i+1)%2:B_spec_dens.shape[0]:2] *= -1 
    
    B_spec_dens[B_spec_dens.shape[0]-1, :] *= - 1. #antiperiodic b.c.
    B_spec_dens[:,B_spec_dens.shape[0]-1] *= - 1. #antiperiodic b.c.

    #identity measure for cont. time prescription
    id_meas = np.zeros(B_spec_dens.shape,dtype=np.complex_)
    for m in range(B_spec_dens.shape[0]//2-1):
        id_meas[2*m+1 , 2*(m+1)] += 1. 
    id_meas[0,B_spec_dens.shape[0]-1] -= 1. 

    id_meas -= id_meas.T
    B_spec_dens += id_meas
    B_spec_dens = linalg.inv(B_spec_dens)#invert the matrix. The result B_spec_dens on the left side defines the IF and corresponds to the matrix A^{-1} from the DMFT guide


#at this point, the matrix B_spec_dens is in the convention with ordering: in_0, out_0, in_1, out_1, ...,in_{M-1},out_{M-1}, i.e. as written in the "guide"



#________________Determine the matrix B_reshuf which can be fed into the routine to compute the correlation matrix_______________

#for trotter scheme (a): to compute the correlation matrix, bring first leg to the last position:
if trotter_convention == 'a':
    B_reshuf = np.zeros((dim_B,dim_B),dtype=np.complex_)
    B_reshuf[:dim_B-1,:dim_B-1] = B_spec_dens[1:dim_B,1:dim_B]
    B_reshuf[dim_B-1,:dim_B-1] = B_spec_dens[0,1:dim_B]
    B_reshuf[:dim_B-1,dim_B-1] = B_spec_dens[1:dim_B,0]
    B_reshuf[dim_B-1,dim_B-1] = B_spec_dens[0,0]

elif trotter_convention == 'b':#for simultaneous impurity-bath evolution, do not reshuffle anything, it's already in the correct order
    B_reshuf = B_spec_dens.copy()
#the matrix B_reshuf can be fed directly into the routine to compute the correlation matrix

#________________Compute correlation matrix which is ready for conversion to MPS_______________________________________
corr_matrix = create_correlation_matrix(B_reshuf)#this is the correlation matrix that should be converted to MPS form. Each 4x4 subblock has the form Lambda_ij = [[<c_i c_j^/dagger> , <c_i c_j> ],[<c_i^/dagger c_j^\dagger> , <c_i/dagger c_j> ]]






###################COMPUTE EXACT PROPAGATOR########################################

#_____________Compute exact non-interacting propagator on the impurity_____________________________________________________________________________
exponent = np.zeros((4*dim_B, 4*dim_B), dtype=np.complex_)#this exponent will contain the exponents of both spin species as well as the impurity dynamics

B = B_spec_dens[::-1,::-1]#this transforms the matrix to the "Grassmann convention", which is a different ordering that is convenient for solving the path integral explicitly
# Influence matices for both spin species
#spin down
exponent[dim_B:2*dim_B, dim_B:2*dim_B] = B[:, :]
#spin up
exponent[2*dim_B:3*dim_B, 2*dim_B:3*dim_B] = B[:, :]


# integration measure
if trotter_convention == 'a':#for successive bath-impurity evolution
    # spin up
    exponent[2*dim_B:3*dim_B-1,1:dim_B] += -np.identity(dim_B-1)
    exponent[3*dim_B-1,0] += -1
    exponent[1:dim_B,2*dim_B:3*dim_B-1] += +np.identity(dim_B-1)
    exponent[0,3*dim_B-1] += +1
    # spin down
    exponent[3*dim_B+1:4*dim_B,dim_B:2*dim_B-1] += -np.identity(dim_B-1)
    exponent[3*dim_B,2*dim_B-1] += -1
    exponent[dim_B:2*dim_B-1,3*dim_B+1:4*dim_B] += +np.identity(dim_B-1)
    exponent[2*dim_B-1,3*dim_B] += +1

elif trotter_convention == 'b': #for simultaneous impurity-bath evolution 
    # spin up
    exponent[2*dim_B:3*dim_B,:dim_B] += - np.identity(dim_B)
    exponent[:dim_B,2*dim_B:3*dim_B] += +np.identity(dim_B)
    # spin down
    exponent[3*dim_B:4*dim_B,dim_B:2*dim_B] += -np.identity(dim_B)
    exponent[dim_B:2*dim_B,3*dim_B:4*dim_B] += +np.identity(dim_B)

#set the impurity gates
T=1-np.tanh(t/2)**2
for i in range(dim_B//2 -1):
    # forward 
    # (matrix elements between up -> down), last factors of (-1) are sign changes to test overlap form
    exponent[dim_B - 2 - 2*i, 4*dim_B - 1 - 2*i] += -1. * np.tanh(t/2) *2/T *np.exp(-1. * mu_up) 
    exponent[dim_B - 1 - 2*i, 4*dim_B - 2 - 2*i] -= -1. * np.tanh(t/2)*2/T *np.exp(-1. * mu_down) 
    #(matrix elements between up -> up)
    exponent[dim_B - 2 - 2*i, dim_B - 1 - 2*i] += 1 *np.cosh(t) *np.exp(-1 * mu_up) *(-1.) 
    #(matrix elements between down -> down)
    exponent[4*dim_B - 2 - 2*i, 4*dim_B - 1 - 2*i] += 1 *np.cosh(t) *np.exp(-1. * mu_down) *(-1.)

    # forward Transpose (antisymm)
    exponent[4*dim_B - 1 - 2*i, dim_B - 2 - 2*i] += 1 * np.tanh(t/2)*2/T *np.exp(-1 * mu_up) 
    exponent[4*dim_B - 2 - 2*i, dim_B - 1 - 2*i] -= 1. * np.tanh(t/2)*2/T *np.exp(-1. * mu_down)
    exponent[dim_B - 1 - 2*i,dim_B - 2 - 2*i] += -1 *np.cosh(t) *np.exp(-1. * mu_up) *(-1.)
    exponent[4*dim_B - 1 - 2*i, 4*dim_B - 2 - 2*i] += -1 *np.cosh(t) *np.exp(-1. * mu_down) *(-1.)

#last application contains antiperiodic bc.:
exponent[0, 3*dim_B +1] += -1. * np.tanh(t/2) *2/T *np.exp(-1. * mu_up) *(-1.) 
exponent[1, 3*dim_B ] -= -1. * np.tanh(t/2)*2/T *np.exp(-1. * mu_down) *(-1.)
#(matrix elements between up -> up)
exponent[0, 1] += 1 *np.cosh(t) *np.exp(-1 * mu_up) *(-1.) *(-1.)
#(matrix elements between down -> down)
exponent[3*dim_B , 3*dim_B + 1] += 1 *np.cosh(t) *np.exp(-1. * mu_down) *(-1.) *(-1.)

# forward Transpose (antisymm)
exponent[3*dim_B +1,0] += 1 * np.tanh(t/2)*2/T *np.exp(-1 * mu_up) *(-1.) 
exponent[3*dim_B,1] -= 1. * np.tanh(t/2)*2/T *np.exp(-1. * mu_down) *(-1.)
exponent[1,0] += -1 *np.cosh(t) *np.exp(-1. * mu_up) *(-1.) *(-1.)
exponent[3*dim_B + 1,3*dim_B] += -1 *np.cosh(t) *np.exp(-1. * mu_down) *(-1.) *(-1.)

   
exponent_inv = linalg.inv(exponent)#this is the matrix whose elements yield the propagator

#________store propagator in hdf5 format_________
filename = '/Users/julianthoenniss/Documents/PhD/data/B_imag_specdens'
with h5py.File(filename + "_propag_GM_dim_B={}_trotter_{}.hdf5".format(dim_B,trotter_convention), 'w') as f:
    #store exponent for benchmark 
    dset_B = f.create_dataset('B', ((dim_B,dim_B)),dtype=np.complex_)
    dset_B[:,:] = B_spec_dens[:,:] #store the matrix B in the convention that is given in the DMFT guide (with the first leg moved to the last position for case of succesive evolution of impurity and bath)

    #store propagators
    dset_propag = f.create_dataset('propag', ((2,dim_B//2)),dtype=np.complex_)

    for tau in range (0,dim_B//2):
        if trotter_convention == 'a':#for successive impurity-bath evolution
            #spin up
            dset_propag[0,tau] = pf.pfaffian(np.array(pd.DataFrame(exponent_inv.T).iloc[[0,3*dim_B -1 -2*tau], [0,3*dim_B -1 -2*tau]]))
            #spin down
            dset_propag[1,tau] = pf.pfaffian(np.array(pd.DataFrame(exponent_inv.T).iloc[[2*dim_B -1 -2*tau,3*dim_B], [2*dim_B -1 -2*tau,3*dim_B]]))
        elif trotter_convention == 'b':#for simultaneous impurity-bath evolution
            #spin up
            dset_propag[0,tau] = pf.pfaffian(np.array(pd.DataFrame(exponent_inv.T).iloc[[0,dim_B -1 -2*tau], [0,dim_B -1 -2*tau]]))
            #spin down
            dset_propag[1,tau] = pf.pfaffian(np.array(pd.DataFrame(exponent_inv.T).iloc[[3*dim_B,4*dim_B-1-2*tau], [3*dim_B,4*dim_B-1-2*tau]]))

            #for mode 'c', test if propagator directly computed from matrix B gives the same as overlap if impurity is set to zero. For this comment out the elimination of overlap-contants
            #dset_propag[1,tau] = np.sqrt(linalg.det(np.array(pd.DataFrame(B_spec_dens.T).iloc[[2*tau,dim_B-1], [2*tau,dim_B-1]])))





