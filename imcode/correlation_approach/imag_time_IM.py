import numpy as np
from scipy import linalg
import h5py
import pandas as pd
import scipy.integrate as integrate
from pfapack import pfaffian as pf
from imag_time_IM_funcs import g_lesser, g_greater, spec_dens, create_correlation_matrix
np.set_printoptions(suppress=False, linewidth=np.nan)

#___________Set parameters_______________________________________________________________________________________
global_gamma = 1.#global energyscale -> set to 1 
beta = 4./global_gamma# here, twice the value than Benedikt (?)
Gamma = 1.



nbr_steps = 2#this is the number of Floquet steps that are performed on the level of MPS
T_ren = 4#this is the number of "substeps" into which one time-step of the environment evolution is subdivided. The result becomes the exact continuous-time result when this parameters is large. This is only meaningful for method "a", i.e. for successive environment-impurity evolution

dim_B = 2 * nbr_steps # Size of exponent matrix in the IF. The variable dim_B is equal to 2*nbr_Floquet_layers
dim_B_temp = dim_B * T_ren
delta_t = beta / (dim_B_temp/2 )#timestep

#for spec-dens matrix (written in convention of the "guide")
int_lim_low = -1.#integration limits for spectral density
int_lim_up = 1.

#trotter convention:
trotter_convention = 'a' #choose between 'a' (for successive evolution of bath and impurity, see DMFT-guide) or 'b' for the simultaneous impurity-bath evolution

alpha = 1 #set this to a values in the interval [0,1]
if trotter_convention == 'b':#for simultaneous evolution, set alpha = 1
    alpha = 1.
    T_ren = 1
    dim_B_temp = dim_B
    delta_t = beta / (dim_B_temp/2 )#timestep

#spin hopping parameter (set to 0 for Anderson impurity model)
t = 0. * (beta / nbr_steps)
#local (spin-dependent) onsite_energies
mu_up =0.3 * (beta / nbr_steps)
mu_down =0.5 * (beta / nbr_steps)

###################COMPUTE INFLUENCE FUNCTIONAL AND ITS CORRELATION MATRIX########################################

#________________Compute the matrix B_spec_dens which defines the Influence functional (this corresponds to the equations in this DMFT guide)_____________________________________________________
B_spec_dens = np.zeros((dim_B_temp,dim_B_temp),dtype=np.complex_)

#___________________for IF defined from a spectral density____________________
for m in range (B_spec_dens.shape[0]//2):
    tau = m * delta_t
    for n in range (m+1,B_spec_dens.shape[0]//2):
        tau_p = n * delta_t
        B_spec_dens[2*m, 2*n + 1] += - (-1.) * delta_t**2 * integrate.quad(lambda x: 1./(2*np.pi) * spec_dens(global_gamma,x) * g_greater(x,beta,tau_p,tau+delta_t* alpha), int_lim_low, int_lim_up)[0]
        B_spec_dens[2*m+1, 2*n ] += - delta_t**2 * integrate.quad(lambda x: 1./(2*np.pi) * spec_dens(global_gamma,x) * g_lesser(x,beta,tau,tau_p+delta_t* alpha), int_lim_low, int_lim_up)[0]
    B_spec_dens[2*m, 2*m + 1] += - (-1.) * delta_t**2 * integrate.quad(lambda x: 1./(2*np.pi) * spec_dens(global_gamma,x) * g_lesser(x,beta,tau,tau+delta_t* alpha), int_lim_low, int_lim_up)[0]
    B_spec_dens[2*m, 2*m + 1] += - 1. #constant from overlap at m=n
    
B_spec_dens -= B_spec_dens.T#antisymmetrize. This is twice the exponent matrix (this is assumed for later part in the code)


"""#___________________for IF defined from a hybridization function (given as vector____________________
B_spec_dens = np.zeros((dim_B_temp,dim_B_temp))
#here, as an example we reproduce the example of the spectral density-result by first defining the hybridization vector:
hyb = np.zeros(dim_B_temp//2)#this vector is the vector coming out of the DMFT loop with hyb[0] corresponding to tau=0, hyb[1] corresponding to tau=1, and so on 

#for our example, initialize hyb[] with the bath greens function as derived in the notes
for n in range (2):
    hyb[n] = - delta_t**2*integrate.quad(lambda x: 1./(2*np.pi) * spec_dens(global_gamma,x) * g_greater(x,beta,(T_ren*nbr_steps + (n-1-alpha))*delta_t, 0), int_lim_low, int_lim_up)[0]
for n in range (2,dim_B_temp//2):
    hyb[n] = delta_t**2*integrate.quad(lambda x: 1./(2*np.pi) * spec_dens(global_gamma,x) * g_greater(x,beta,(n-1-alpha)*delta_t, 0), int_lim_low, int_lim_up)[0]
hyb = np.append(hyb[1:],-hyb[0])#reshuffle first element to last position with negative sign, such that matrix an be initialized easier

for m in range (B_spec_dens.shape[0]//2):
    B_spec_dens[2*m,2*m+1::2] = hyb[:len(hyb)-m]
    B_spec_dens[2*m+1,2*m+2::2] = hyb[len(hyb)-1:m:-1]
    B_spec_dens[2*m, 2*m + 1] += - 1. #constant from overlap at m=n
B_spec_dens -= B_spec_dens.T#antisymmetrize. This is twice the exponent matrix (this is assumed for later part in the code)
"""

if trotter_convention == 'a' and T_ren > 1: 
    #add intermediate integration measure to integrate out internal legs
    for i in range (dim_B//2 ):
        for j in range (T_ren-1):
            B_spec_dens[2*i*T_ren + 1 + 2*j,2*i*T_ren+2+ 2*j] += -1  
            B_spec_dens[2*i*T_ren+2+ 2*j,2*i*T_ren + 1 + 2*j] += 1  

    #select submatrix that contains all intermediate times that are integrated out
    B_spec_dens_sub =  np.zeros((dim_B_temp - dim_B, dim_B_temp - dim_B),dtype=np.complex_)
    for i in range (dim_B//2 ):
        for j in range (dim_B//2 ):
            B_spec_dens_sub[i*(2*T_ren-2):i*(2*T_ren-2 )+2*T_ren-2,j*(2*T_ren-2):j*(2*T_ren-2 )+2*T_ren-2] = B_spec_dens[2*i*T_ren+1:2*(i*T_ren + T_ren)-1,2*j*T_ren+1:2*(j*T_ren + T_ren)-1]

    #matrix coupling external legs to integrated (internal) legs
    B_spec_dens_coupl =  np.zeros((dim_B_temp - dim_B,dim_B),dtype=np.complex_)
    for i in range (dim_B//2 ):
        for j in range (dim_B//2 ):
            B_spec_dens_coupl[i*(2*T_ren-2):i*(2*T_ren-2 )+2*T_ren-2,2*j] = B_spec_dens[2*i*T_ren+1:2*(i*T_ren + T_ren)-1,2*j*T_ren]
            B_spec_dens_coupl[i*(2*T_ren-2):i*(2*T_ren-2 )+2*T_ren-2,2*j+1] = B_spec_dens[2*i*T_ren+1:2*(i*T_ren + T_ren)-1,2*(j+1)*T_ren-1]

    #part of matriy that is neither integrated nor coupled to integrated variables
    B_spec_dens_ext = np.zeros((dim_B,dim_B),dtype=np.complex_)
    for i in range (dim_B//2 ):
        for j in range (dim_B//2 ):
            B_spec_dens_ext[2*i,2*j] = B_spec_dens[2*i*T_ren,2*j*T_ren]
            B_spec_dens_ext[2*i+1,2*j] = B_spec_dens[2*(i+1)*T_ren-1,2*j*T_ren]
            B_spec_dens_ext[2*i,2*j+1] = B_spec_dens[2*i*T_ren,2*(j+1)*T_ren-1]
            B_spec_dens_ext[2*i+1,2*j+1] = B_spec_dens[2*(i+1)*T_ren-1,2*(j+1)*T_ren-1]

    B_spec_dens = B_spec_dens_ext + B_spec_dens_coupl.T @ linalg.inv(B_spec_dens_sub) @ B_spec_dens_coupl



elif trotter_convention == 'b':#for simultaneous impurity-bath evolution. Note that the input here is the matrix B_specdens, computed with successive-application-trotter-scheme  
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
with h5py.File(filename + "_propag_GM_steps={}_trotter_{}_T_ren={}.hdf5".format(nbr_steps,trotter_convention,int(T_ren)), 'w') as f:
    #store exponent for benchmark 
    dset_B = f.create_dataset('B', ((dim_B,dim_B)),dtype=np.complex_)
    dset_B[:,:] = B_spec_dens[:,:] #store the matrix B in the convention that is given in the DMFT guide (with the first leg moved to the last position for case of succesive evolution of impurity and bath)

    #store propagators
    dset_propag = f.create_dataset('propag', ((2,dim_B//2+1)),dtype=np.complex_)
    dset_times = f.create_dataset('times', ((dim_B//2+1,)),dtype=np.float_)

    for tau in range (0,dim_B//2):
        dset_times[tau] = delta_t * T_ren * tau
        if trotter_convention == 'a':#for successive impurity-bath evolution
            #spin up
            dset_propag[0,tau] = pf.pfaffian(np.array(pd.DataFrame(exponent_inv.T).iloc[[0,3*dim_B -1 -2*tau], [0,3*dim_B -1 -2*tau]]))
            #spin down
            dset_propag[1,tau] = pf.pfaffian(np.array(pd.DataFrame(exponent_inv.T).iloc[[2*dim_B -1 -2*tau,3*dim_B], [2*dim_B -1 -2*tau,3*dim_B]]))
            print('up',dset_propag[0,tau])
            print('down',dset_propag[1,tau])
        elif trotter_convention == 'b':#for simultaneous impurity-bath evolution
            #spin up
            dset_propag[0,tau] = pf.pfaffian(np.array(pd.DataFrame(exponent_inv.T).iloc[[0,dim_B -1 -2*tau], [0,dim_B -1 -2*tau]]))
            #spin down
            dset_propag[1,tau] = pf.pfaffian(np.array(pd.DataFrame(exponent_inv.T).iloc[[3*dim_B,4*dim_B-1-2*tau], [3*dim_B,4*dim_B-1-2*tau]]))
    
    dset_times[dim_B//2] = delta_t * T_ren * dim_B//2
    dset_propag[0,dim_B//2] = 1 - dset_propag[0,0]
    dset_propag[1,dim_B//2] = 1 - dset_propag[1,0]
            #for mode 'c', test if propagator directly computed from matrix B gives the same as overlap if impurity is set to zero. For this comment out the elimination of overlap-contants
            #dset_propag[1,tau] = np.sqrt(linalg.det(np.array(pd.DataFrame(B_spec_dens.T).iloc[[2*tau,dim_B-1], [2*tau,dim_B-1]])))





