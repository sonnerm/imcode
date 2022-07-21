
import numpy as np
from scipy import linalg
import sys
from create_correlation_block import create_correlation_block
from entropy import entropy
import h5py

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=470)

conv = 'M' # 'J': my convention, 'M': Michael's convention
conv_out = 'M' # 'J': my convention, 'M': Michael's convention

B_l = []
B_r = [] 
filename_l = '/Users/julianthoenniss/Documents/PhD/data/Millis_mu=-.2_timestep=0.05_T=200_hyb=0.05_D=1_Michaels_conv'
filename_r = '/Users/julianthoenniss/Documents/PhD/data/Millis_mu=.2_timestep=0.05_T=200_hyb=0.05_D=1_Michaels_conv'
filename = '/Users/julianthoenniss/Documents/PhD/data/Millis_interleaved_timestep=0.05_hyb=0.05_T=200_D=1'
#filename_l = '/Users/julianthoenniss/Documents/PhD/data/compmode=C_o=2_Jx=1.0_Jy=1.0_g=0.0mu=-2.5_del_t=0.1_beta=200.0_L=200_init=4'
#filename_r = '/Users/julianthoenniss/Documents/PhD/data/compmode=C_o=1_Jx=1.0_Jy=1.0_g=0.0mu=2.5_del_t=0.1_beta=200.0_L=200_init=4'
#filename = '/Users/julianthoenniss/Documents/PhD/data/XX_deltamu=5.0'

if conv_out == 'J':
    filename += '_my_conv' 
    print('storing result in Js convention')
elif conv_out == 'M':
    filename += '_Michaels_conv' 
    print('storing result in Ms convention')

max_time1 = 201#maximal floquet_nbr_steps to set matrix stoage to correct size
iterations = 1

with h5py.File(filename + ".hdf5", 'w') as f:
        dset_temp_entr = f.create_dataset('temp_entr', (iterations, max_time1),dtype=np.float_)
        dset_IM_exponent = f.create_dataset('IM_exponent', (iterations, 4 * max_time1 , 4 * max_time1),dtype=np.complex_)
        dset_times = f.create_dataset('times', (iterations,),dtype=np.int16)
for iter in range (0,iterations):
    
    #if exponent B is read out
    with h5py.File(filename_l + '.hdf5', 'r') as f:
        #times_read = f['temp_entr']
        times_read = f['times']
        nbr_Floquet_layers  = int(times_read[iter])
        print('times: ', nbr_Floquet_layers, ' iteration: ', iter )
    
    
    
    #else:
    #nbr_Floquet_layers = iter

    with h5py.File(filename_l + '.hdf5', 'r') as f:
        print(4*nbr_Floquet_layers,4*nbr_Floquet_layers)
        B_l = f['IM_exponent'][iter,:4*nbr_Floquet_layers,:4*nbr_Floquet_layers]
        #print(f['IM_exponent'][iter,:4*(nbr_Floquet_layers+1),:4*(nbr_Floquet_layers+1)])
    with h5py.File(filename_r + '.hdf5', 'r') as f:
        print(4*nbr_Floquet_layers,4*nbr_Floquet_layers)
        B_r = f['IM_exponent'][iter,:4*nbr_Floquet_layers,:4*nbr_Floquet_layers]

    B_r_orig = B_r
    dim_B = B_l.shape[0]

    #rotate into correct basis with input/output variables as specified for Michael

    S = np.zeros(B_l.shape,dtype=np.complex_)
    for i in range (dim_B//4):#order plus and minus next to each other
        S [dim_B // 2 - (2 * i) - 2,4 * i] = 1
        S [dim_B // 2 - (2 * i) - 1,4 * i + 2] = 1
        S [dim_B // 2 + (2 * i) ,4 * i + 1] = 1
        S [dim_B // 2 + (2 * i) + 1,4 * i + 3] = 1

    #the following two transformation bring it into in/out- basis (not theta, zeta)
    rot = np.zeros(B_l.shape,dtype=np.complex_)
    for i in range(0,dim_B, 2):#go from bar, nonbar to zeta, theta
        rot[i,i] = 1./np.sqrt(2)
        rot[i,i+1] = 1./np.sqrt(2)
        rot[i+1,i] = - 1./np.sqrt(2) * np.sign(dim_B//2 - i-1)
        rot[i+1,i+1] = 1./np.sqrt(2) * np.sign(dim_B//2 - i-1)

    U = np.zeros(B_l.shape,dtype=np.complex_)#order in the way specified in pdf for Michael (forward,backward,forward,backward,...)
    for i in range (dim_B//4):
        U[4*i, dim_B //2 - (2*i) -1] = 1
        U[4*i + 1, dim_B //2 + (2*i)] = 1
        U[4*i + 2, dim_B //2 - (2*i) -2] = 1
        U[4*i + 3, dim_B //2 + (2*i) + 1] = 1
                
    if conv == 'J':
    #print(U.shape, rot.shape, S.shape,B_l.shape)
        B_l = U @ rot.T @ S @ B_l @ S.T @ rot @ U.T 
        B_r = U @ rot.T @ S @ B_r @ S.T @ rot @ U.T 


    #matrix containing all variables not integrated over (output left and input right)
    C = np.zeros(B_l.shape,dtype=np.complex_)
    for i in range (0,dim_B//4):
        for j in range (0,dim_B//4):
            C[2*i:2*i+2, 2*j:2*j+2] = B_r[4*i:4*i+2,4*j:4*j+2]
            C[2*i + dim_B//2 :2*i + dim_B//2 +2, 2*j + dim_B//2 :2*j + dim_B//2 +2] =  B_l[4*i+2:4*i+4,4*j+2:4*j+4]
    print('C')
    #print(np.real(C))
    #matrix coupling the non-integrated variables (output left and input right) to the integrated ones 
    R = np.zeros(B_l.shape,dtype=np.complex_)
    for i in range (0,dim_B//4):
        for j in range (0,dim_B//4):
            R[2*i:2*i+2, 2*j:2*j+2] = B_r[4*i:4*i+2,4*j+2:4*j+4]
            R[2*i + dim_B//2 :2*i + dim_B//2 +2, 2*j + dim_B//2 :2*j + dim_B//2 +2] = B_l[4*i+2:4*i+4,4*j:4*j+2]
    print('R')
    #print(np.real(R))
    #entries that are quadratic in integration variables
    A = np.zeros(B_l.shape,dtype=np.complex_)
    for i in range (0,dim_B//4):
        for j in range (0,dim_B//4):
            A[2*i:2*i+2, 2*j:2*j+2] = B_r[4*i+2:4*i+4,4*j+2:4*j+4]
            A[2*i + dim_B//2 :2*i + dim_B//2 +2, 2*j + dim_B//2 :2*j + dim_B//2 +2] = B_l[4*i:4*i+2,4*j:4*j+2]

    #Grassmann integration measure:
    for i in range (0,dim_B//4):      
        A[2*i, 2*i+dim_B//2] -= 1
        A[2*i+1, 2*i+dim_B//2+1] += 1
        A[2*i+dim_B//2, 2*i] += 1
        A[2*i+dim_B//2+1, 2*i+1] -= 1

    print('A')
    #print(np.real(A))
    #exponent of joined IM
    B_joined = C + R @ linalg.inv(A) @ R.T

    B_joined_reshuf = np.zeros(B_joined.shape,dtype=np.complex_)
    #reshuffle in the way specified in pdf for Michael (forward-in,backward-in,forward-out,backward-out,...)
    for i in range (0,dim_B//4):
        for j in range (0,dim_B//4):
            B_joined_reshuf[4*i : 4*i+2 , 4*j : 4*j+2] = B_joined[2*i : 2*i+2 , 2*j : 2*j+2]
            B_joined_reshuf[4*i+2 : 4*i+4 , 4*j+2 : 4*j+4] = B_joined[2*i +  dim_B//2: 2*i+2 +  dim_B//2, 2*j+  dim_B//2 : 2*j+2+  dim_B//2]
            B_joined_reshuf[4*i : 4*i+2 , 4*j+2 : 4*j+4] = B_joined[2*i : 2*i+2 , 2*j+  dim_B//2 : 2*j+  dim_B//2 +2]
            B_joined_reshuf[4*i+2 : 4*i+4 , 4*j : 4*j+2] = B_joined[2*i +  dim_B//2: 2*i+2 +  dim_B//2, 2*j : 2*j+2]

    
    
    if conv_out == 'J':#This rotates aways from Michael's basis into theta/zeta-basis
        B_joined_reshuf = S.T @ rot @  U.T @ B_joined_reshuf @ U @ rot.T @ S



    
    with h5py.File(filename + '.hdf5', 'a') as f:
        IM_data = f['IM_exponent']
        IM_data[iter,:B_joined_reshuf.shape[0],:B_joined_reshuf.shape[0]] = B_joined_reshuf[:,:]
        times_data = f['times']
        times_data[iter] = nbr_Floquet_layers
    
    correlation_block = create_correlation_block(B_joined_reshuf, nbr_Floquet_layers, filename)
    time_cuts = np.arange(1, nbr_Floquet_layers)
    
    #with h5py.File(filename + '.hdf5', 'a') as f:
    #    entr_data = f['temp_entr']
    #    entr_data[iter,0] = nbr_Floquet_layers
    #    for cut in time_cuts:
    #        #print('calculating entropy at time cut:', cut)
    #        entr_data[iter,cut] = float(entropy('C', correlation_block, nbr_Floquet_layers, cut, iter, filename))
    