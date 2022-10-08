from Lohschmidt import Lohschmidt
from evolution_matrix import evolution_matrix
from compute_generators import compute_generators
import numpy as np
import h5py
from create_correlation_block import create_correlation_block
from entropy import entropy
from IM_exponent import IM_exponent
from add_cmplx_random_antisym import add_cmplx_random_antisym
from ham_gs import create_exact_Floquet_ham
from plot_entropy import plot_entropy
from matrix_diag import matrix_diag
from scipy.linalg import expm, logm
from scipy import linalg
from dress_density_matrix import dress_density_matrix
from compute_generators import compute_generators
from Lohschmidt import Lohschmidt
from ising_gamma import ising_gamma
import sys
from gm_integral import gm_integral
import os
import math



np.set_printoptions(linewidth=np.nan, precision=1, suppress=True)

# define fixed parameters:
# step sizes for total times t
time_0 = 300
max_time1 =301
max_time2 = 301
stepsize1 = 1
stepsize2 = 1

order = 2#order = 1: interaction layer/odd first, order = 2: even layer first
time_array = np.append(np.arange(time_0, max_time1, stepsize1) , np.arange(max_time1, max_time2, stepsize2))
print(time_array)

mode =  sys.argv[1] # 'G': compute temporal entanglement entropy from Grassmann approach, 'C': compute temporal entanglement entropy from correlation approach, 'L': compute Lohschmidt echo
write_mode = sys.argv[2] #if the argument is 'w', overwrite file if it exists, otherwise append if it exists
# lattice sites (in the environment):
nsites = int(sys.argv[3])
# model parameters:
del_t = float(sys.argv[4])
Jx = float(sys.argv[5]) * del_t #* np.pi/2 #0.5# 0.31 # 0.31
Jy =float(sys.argv[6])* del_t# * np.pi/2#np.pi/4+0.3#np.pi/4
g =float(sys.argv[7])* del_t #* np.pi/2#np.pi/4+0.3
init_state = int(sys.argv[8])#0: thermal e^{-\beta XX}, 1: Bell pairs, 2: BCS_GS, 3: Inf. Temp.. Invalied entries will be set to Inf. Temp. (=3)
beta = float(sys.argv[9])#0.4  # temperature
mu_initial_state = float(sys.argv[10])
if mode == 'L':
    g_boundary_mag = float(sys.argv[11])

print('mode', mode)
print('write_mode', write_mode)
print('nsites', nsites)
print('del_t', del_t)
print('Jx', Jx)
print('Jy', Jy)
print('g', g)
print('beta', beta)
print('init_state', init_state)
print('mu_init_state', mu_initial_state)
if mode == 'L':
    print('g_boundary_mag', g_boundary_mag)

coupling_strength = 0.3162
Jx_coupling = Jx * coupling_strength
Jy_coupling = Jy * coupling_strength


potentials = [0.0,0.005,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.055,0.06,0.07]

for potential in potentials:
    iterator = 0 # iterator at beginning of job. If file exist and job is continued, this will be set to the appropriate value below
    # total_time = 0 means one floquet-layer
    mu_initial_state = - potential /2
    #set location for data storage
    mac_path = '/Users/julianthoenniss/Documents/Studium/PhD/data/correlation_approach/'
    work_path = '/Users/julianthoenniss/Documents/PhD/data/'
    fiteo1_path = '/home/thoennis/data/correlation_approach/'
    baobab_path = '/home/users/t/thoennis/scratch/'

    filename = work_path + 'compmode=' + str(mode)+ '_o=' + str(order) + '_Jx=' + str(Jx/del_t) + '_Jy=' + str(Jy/del_t) + '_g=' + str(g/del_t) + 'mu=' + str(mu_initial_state) +'_del_t=' + str(del_t)+ '_beta=' + str(beta)+ '_L=' + str(nsites) + '_init=' + str(init_state) +  '_coupling=' + str(coupling_strength)
    if mode == 'L':
        filename += '_g_boundary_mag=' + str(g_boundary_mag) 

    print('filename:', filename)


    if os.path.isfile(filename+".hdf5") and write_mode != 'w':
        with h5py.File(filename + '.hdf5', 'r') as f:
            crit = True
            print(f['temp_entr'][:,0])
            while crit:
                crit = (f['temp_entr'][iterator,0] > 0)
                iterator += 1
            iterator -= 1
        print('File exists: continuing from iteration' , iterator, '..')

    else:
        print('Create/Truncate File..')
        with h5py.File(filename + ".hdf5", 'w') as f:
            dset_temp_entr = f.create_dataset('temp_entr', ((max_time1-time_0)//stepsize1 + (max_time2- max_time1)//stepsize2 + 1, max_time2 + stepsize2),dtype=np.float_)
            dset_entangl_specrt = f.create_dataset('entangl_spectr', (int((max_time1-time_0)/stepsize1) + (max_time2- max_time1)//stepsize2 + 1, max_time2 + stepsize2, 8*(max_time2 + stepsize2)),dtype=np.float_)
            dset_IM_exponent = f.create_dataset('IM_exponent', (int((max_time1-time_0)/stepsize1) + (max_time2- max_time1)//stepsize2 + 1, 4 * (max_time2 + stepsize2), 4 * (max_time2 + stepsize2)),dtype=np.complex_)
            dset_edge_corr = f.create_dataset('edge_corr', (int((max_time1-time_0)/stepsize1) + (max_time2- max_time1)//stepsize2 + 1, 2 * (4*(max_time2 + stepsize2) - 1), 2 * (4*(max_time2 + stepsize2) - 1) ),dtype=np.complex_)
            #dset = f.create_dataset('bulk_corr', (int(max_time1-time_0)/stepsize1) + (max_time2- max_time1)//stepsize2 + 1, 2 * (4*(max_time2 + stepsize2) - 1) *nsites, 2 * (4*(max_time2 + stepsize2) - 1) *nsites),dtype=np.complex_)
            dset_init_BCS_state = f.create_dataset('init_BCS_state', (nsites, nsites),dtype=np.complex_)
            dset_const_blip = f.create_dataset('const_blip', ((max_time1-time_0)//stepsize1 + (max_time2- max_time1)//stepsize2 + 1,),dtype=np.float_)

    if mode == "C":

        if (abs((abs(Jx-Jy)*2/np.pi)%2 - 1)< 1.e-8):#if difference between Jx and Jy is close to multiples of pi/2
            beta_tilde = 0
        else:
            beta_tilde = 0.5 * np.log( (1 + np.tan(Jx_coupling) * np.tan(Jy_coupling) ) / (1 - np.tan(Jx_coupling) * np.tan(Jy_coupling) ) + 0.j) 
            #beta_tilde = - 0.5 * np.log( (1 + np.tan(Jx-np.pi/2) * np.tan(Jy) ) / (1 - np.tan(Jx-np.pi/2) * np.tan(Jy) ) + 0.j) 
        #print('beta_tilde', beta_tilde, beta_tilde_comp, np.exp(beta_tilde))
        # define initial density matrix and determine matrix which diagonalizes it:
        # this is the EXPONENT of the BARE gaussian density matrix
        rho_0_exponent = np.zeros((2 * nsites, 2 * nsites), dtype=np.complex_)
        # must be initialized as matrix that diagonalizes dressed density matrix
        N_t = np.identity(2 * nsites, dtype=np.complex_)
        # find generators and matrices which diagonalize the composite Floquet operator:
        G_XY_even, G_XY_odd, G_g, G_1 = compute_generators(nsites, Jx, Jy, g, beta_tilde)
        evolution_matrix_result, F_E_prime, F_E_prime_dagger= evolution_matrix(nsites, G_XY_even, G_XY_odd, G_g, G_1) 

        M, M_E, eigenvalues_G_eff, eigenvalues_G_eff_E, G_eff_E = matrix_diag(nsites, G_XY_even, G_XY_odd, G_g, G_1, order, Jx, Jy, g,)#order = 1: interaction layer/odd first, order = 2: even layer first
    for nbr_Floquet_layers in time_array[iterator:]:
        mode =  sys.argv[1]
        N_sites_needed_for_entr = nsites#2*nbr_Floquet_layers 

        #store array with times in entropy-dataset, regardless of whether entropy is acutally computed (also used for Lohschmidt)
        with h5py.File(filename + '.hdf5', 'a') as f:
            entr_data = f['temp_entr']
            entr_data[iterator,0] = nbr_Floquet_layers#write array with times
        

        if mode == "C" or mode == "G":
            if mode == "C":
                n_expect, N_t, Lambda = dress_density_matrix(rho_0_exponent, F_E_prime, F_E_prime_dagger,M,M_E,  eigenvalues_G_eff, eigenvalues_G_eff_E, beta_tilde, nbr_Floquet_layers, init_state, G_XY_even,G_XY_odd, order, beta, mu_initial_state,G_eff_E,Jx,Jy)#order = 1: interaction layer/odd first, order = 2: even layer first
                B = IM_exponent(evolution_matrix_result, N_t, nsites,nbr_Floquet_layers, Jx_coupling, Jy_coupling, beta_tilde, n_expect)
                
                
                
                #for production mode, turn of ALL of the below rotation. They bring the 'C'-exponent (correlation approach) into the basis of the 'G'-exponent (Grassmann) approach). For Michael, turn them on!
                """
                S = np.zeros(B.shape,dtype=np.complex_)
                for i in range (nbr_Floquet_layers):#order plus and minus next to each other
                    S [B.shape[0] // 2 - (2 * i) - 2,4 * i] = 1
                    S [B.shape[0] // 2 - (2 * i) - 1,4 * i + 2] = 1
                    S [B.shape[0] // 2 + (2 * i) ,4 * i + 1] = 1
                    S [B.shape[0] // 2 + (2 * i) + 1,4 * i + 3] = 1
                
                B = S @ B @ S.T

                #the following two transformation bring it into in/out- basis (not theta, zeta)
                rot = np.zeros((4 * nbr_Floquet_layers,4 * nbr_Floquet_layers))
                for i in range(0,4*nbr_Floquet_layers, 2):#go from bar, nonbar to zeta, theta
                    rot[i,i] = 1./np.sqrt(2)
                    rot[i,i+1] = 1./np.sqrt(2)
                    rot[i+1,i] = - 1./np.sqrt(2) * np.sign(2*nbr_Floquet_layers - i-1)
                    rot[i+1,i+1] = 1./np.sqrt(2) * np.sign(2*nbr_Floquet_layers - i-1)
                B = rot.T @ B @ rot
                """
                

            else:
                B = gm_integral(init_state, Jx,Jy,g,mu_initial_state, beta, N_sites_needed_for_entr,nbr_Floquet_layers, filename, iterator)
                #the following block expresses B in the same basis as in the correlation approach such that the two matrices can directly be compared
                
                #the following two rotations (rot and S) bring the 'G'-exponent (Grassmann approach) into the basis of the 'C'-exponent (correlation approach). For Michael, turn them off!
                rot = np.zeros((4 * nbr_Floquet_layers,4 * nbr_Floquet_layers))
                for i in range(0,4*nbr_Floquet_layers, 2):#go from bar, nonbar to zeta, theta
                    rot[i,i] = 1./np.sqrt(2)
                    rot[i,i+1] = 1./np.sqrt(2)
                    rot[i+1,i] = - 1./np.sqrt(2)#* np.sign(2*nbr_Floquet_layers - i-1)#the commented part should be activated in order to compare to correaltion-mode IM exponent (uses different sign for convention of theta-grassmann fiel on forward and backwar branch)
                    rot[i+1,i+1] = 1./np.sqrt(2) #* np.sign(2*nbr_Floquet_layers - i-1)
                B = rot @ B @ rot.T
                
                S = np.zeros(B.shape,dtype=np.complex_)
                for i in range (nbr_Floquet_layers):#order plus and minus next to each other
                    S [B.shape[0] // 2 - (2 * i) - 2,4 * i] = 1
                    S [B.shape[0] // 2 - (2 * i) - 1,4 * i + 2] = 1
                    S [B.shape[0] // 2 + (2 * i) ,4 * i + 1] = 1
                    S [B.shape[0] // 2 + (2 * i) + 1,4 * i + 3] = 1
                B = S.T @ B @ S
                mode = 'C'
                
            with h5py.File(filename + '.hdf5', 'a') as f:
                IM_data = f['IM_exponent']
                IM_data[iterator,:B.shape[0],:B.shape[0]] = B[:,:]
            print('Finished writing data at iteration', iterator)

            """
            #for Michael:
            filename_corr = filename + '_Michael_convention'
            #for production mode, turn of ALL of the below rotation. They bring the 'C'-exponent (correlation approach) into the basis of the 'G'-exponent (Grassmann) approach). For Michael, turn them on!
            S = np.zeros(B.shape,dtype=np.complex_)
            for i in range (nbr_Floquet_layers):#order plus and minus next to each other
                S [B.shape[0] // 2 - (2 * i) - 2,4 * i] = 1
                S [B.shape[0] // 2 - (2 * i) - 1,4 * i + 2] = 1
                S [B.shape[0] // 2 + (2 * i) ,4 * i + 1] = 1
                S [B.shape[0] // 2 + (2 * i) + 1,4 * i + 3] = 1
            
            B = S @ B @ S.T

            #the following two transformation bring it into in/out- basis (not theta, zeta)
            rot = np.zeros((4 * nbr_Floquet_layers,4 * nbr_Floquet_layers))
            for i in range(0,4*nbr_Floquet_layers, 2):#go from bar, nonbar to zeta, theta
                rot[i,i] = 1./np.sqrt(2)
                rot[i,i+1] = 1./np.sqrt(2)
                rot[i+1,i] = - 1./np.sqrt(2) * np.sign(2*nbr_Floquet_layers - i-1)
                rot[i+1,i+1] = 1./np.sqrt(2) * np.sign(2*nbr_Floquet_layers - i-1)
            B = rot.T @ B @ rot

            #here, the matrix B is in the Grassmann convention
            # adjust signs that make the influence matrix a vectorized state
            #for i in range (B.shape[0]):
            #    for j in range (B.shape[0]):
            #        if (i+j)%2 == 1:
            #            B[i,j] *= -1
            #filename_corr += '_SIGNCHANGED'
            
            U = np.zeros(B.shape)#order in the way specified in pdf for him (forward,backward,forward,backward,...)
            for i in range (nbr_Floquet_layers):
                U[4*i, B.shape[0] //2 - (2*i) -1] = 1
                U[4*i + 1, B.shape[0] //2 + (2*i)] = 1
                U[4*i + 2, B.shape[0] //2 - (2*i) -2] = 1
                U[4*i + 3, B.shape[0] //2 + (2*i) + 1] = 1
            B = U @ B @ U.T 
            

            correlation_block = create_correlation_block(B, nbr_Floquet_layers, filename_corr)
            eigs = linalg.eigvalsh(np.real(correlation_block))
            for element in eigs:
                if abs(1- element) > 1.e-8 and abs(element) > 1.e-8:
                    print('found elemens of value, ', element)
            #print(linalg.eigvalsh(correlation_block))
            time_cuts = np.arange(1, nbr_Floquet_layers)
            
            print('Starting to write data at iteration', iterator)
            with h5py.File(filename + '.hdf5', 'a') as f:
                entr_data = f['temp_entr']
                for cut in time_cuts:
                    #print('calculating entropy at time cut:', cut)
                    entr_data[iterator,cut] = float(entropy(mode, correlation_block, nbr_Floquet_layers, cut, iterator, filename))
            """

            

        elif mode == 'L':
            Lohschmidt(init_state, Jx, Jy,g,beta, mu_initial_state,g_boundary_mag, nsites, nbr_Floquet_layers, filename, iterator)

        else:
            print('No valid operation mode specified.')

        iterator += 1


    with h5py.File(filename + '.hdf5', 'r') as f:
        entr_data = f['temp_entr']
        np.set_printoptions(linewidth=np.nan, precision=7, suppress=True)
        print(entr_data[:])

    #plot_entropy(entropy_values, iterator, Jx/del_t, Jy/del_t, g/del_t, del_t, beta, nsites, filename, ising_gamma_times, ising_gamma_values)
