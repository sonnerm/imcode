from correlator import correlator
from numpy import dtype, linalg
from scipy.linalg import expm
import numpy as np
#check equal-time Majorana self-correlations (correlations < (c + c^dagger)(c + c^dagger) > and <- (c - c^dagger)(c - c^dagger) > at equal times yield partition sum since (c + c^dagger)(c + c^dagger) = -(c - c^dagger)(c - c^dagger) = 1)

def test_identity_correlations(A, n_expect, ntimes):
    status = 0
    print ('Testing identity correlations..')
    for tau in range (ntimes):
        #first Majorana type (0, Theta, -)
        #forward branch
        test = correlator(A,n_expect, 0, 0, tau, 0, 0, tau)
        if (abs(test) - 1) > 1e-6:
            print('Identity correlation test not passed, forward branch, Majorana type 0 (Theta/- Majorana)',' tau=', tau, 'value: ', test) 
            status += 1

        #backward brnach
        test = correlator(A,n_expect, 1, 0, tau, 1, 0, tau)
        if (abs(test) - 1) > 1e-6:
            print('Identity correlation test not passed, backward branch, Majorana type 0 (Theta/- Majorana)',' tau=', tau, 'value: ', test) 
            status += 1


        #second Majorana type (1, Zeta, +)
        #forward branch
        test = correlator(A,n_expect, 0, 1, tau, 0, 1, tau)
        if (abs(test) - 1) > 1e-6:
            print('Identity correlation test not passed, forward branch, Majorana type 1 (Zeta/+ Majorana)',' tau=', tau, 'value: ', test) 
            status += 1
        #backward branch
        test = correlator(A,n_expect, 1, 1, tau, 1, 1, tau)
        if (abs(test) - 1) > 1e-6:
            print('Identity correlation test not passed, backward branch, Majorana type 1 (Zeta/+ Majorana)',' tau=', tau, 'value: ', test) 
            status += 1
    if status == 0:
        print ('Testing identity correlations sucessfully terminated..')    
    else: 
        print ('Identity correlation tests not passed in ',status, 'cases.' )


def anti_sym_check(matrix):
    dim = len(matrix[0])
    check = 0
    for i in range (dim):
        for j in range (i,dim):
            check += abs(matrix[i,j] + matrix[j,i])
    
    if check > 1e-6:
        print ('Antisymmetry test not passed..', check)
    
    else:
        print ('Antisymmetry test successfully passed..')


def evolve_rho(rho,B):
    dim_B = B.shape[0]
    
    #add local impurity channel
    #measure (this imposes the IT initial state)
    for i in range (dim_B//2-1):
        B[2 * i + 1 ,2 * i+2 ] += 1
        B[2 * i + 2 ,2 * i+1 ] -= 1


    A = B[1:dim_B-1,1:dim_B-1]
    R = B[[0,dim_B-1],1:dim_B-1]
    C = np.zeros((2,2),dtype=np.complex_)
    C[0,1] = B[0,dim_B-1]
    C[1,0] = B[dim_B-1,0]

    A_inv = linalg.inv(A)
    rho_exponent_evolved = 0.5*(R @ A_inv @ R.T + C)
    norm = np.sqrt(linalg.det(A))
    return rho_exponent_evolved, norm

def evolve_rho_T(rho,B,T,t):
    dim_B = B.shape[0]
    
    #add local impurity channel
    #measure (this imposes the IT initial state)
    for i in range (dim_B//2-1):
        B[2 * i + 1 ,2 * i+2 ] += 1
        B[2 * i + 2 ,2 * i+1 ] -= 1

    print(B[0:2*(T-t)+1,1 + 2*(T-t) :dim_B-1 - 2*(T-t)].shape)
    print(B[dim_B-1 - 2*(T-t):dim_B,1 + 2*(T-t) :dim_B-1 - 2*(T-t)].shape)
    A = B[1 + 2*(T-t):dim_B-1 - 2*(T-t),1 + 2*(T-t):dim_B-1 - 2*(T-t)]

    R = np.bmat([[B[0:2*(T-t)+1,1 + 2*(T-t) :dim_B-1 - 2*(T-t)]],[B[dim_B-1 - 2*(T-t):dim_B,1 + 2*(T-t) :dim_B-1 - 2*(T-t)]]])

    print(2*(T-t)+1,dim_B-(dim_B-1 - 2*(T-t)))
   
    C = np.bmat([[B[0:2*(T-t)+1,0:2*(T-t)+1], B[0:2*(T-t)+1,dim_B-1 - 2*(T-t):dim_B]],[B[dim_B-1 - 2*(T-t):dim_B,0:2*(T-t)+1], B[dim_B-1 - 2*(T-t):dim_B,dim_B-1 - 2*(T-t):dim_B]]])
 
    print('Rshape',R.shape)
    print('Ashape',A.shape)
    print('Cshape',C.shape)
    A_inv = linalg.inv(A)
    rho_exponent_evolved = 0.5*(R @ A_inv @ R.T + C ) 
    rho_exponent_evolved_result = np.zeros((2,2),dtype=np.complex_)
    rho_exponent_evolved_result[0,1] = rho_exponent_evolved[2*(T-t),C.shape[0] - 1 - 2*(T-t)]
    rho_exponent_evolved_result[1,0] = rho_exponent_evolved[C.shape[0] - 1 - 2*(T-t), 2*(T-t)]
    norm = np.sqrt(linalg.det(A))
    return rho_exponent_evolved_result, norm 