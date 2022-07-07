from re import sub
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from scipy import sparse
import h5py

np.set_printoptions(suppress=False, linewidth=np.nan)


def determine_U(sub_corr,eigvec, lower, upper, total_dim):
    #lower = 2
    #upper = 8
    #total_dim = 12
    #sub_corr = np.random.rand(6,6)
    #sub_dim = len(sub_corr[0])
    #eigvals, eigvecs = linalg.eigh(sub_corr)
    #eigvec = eigvecs[0]

    U_sub = np.identity(len(sub_corr))
    sub_dim = upper - lower

    eigvec_temp = eigvec

    for prefac in range (2):
        for i in range (0,sub_dim - 3 - prefac, 2):
            exp_phi = 1
            theta = 0
            if (abs(eigvec_temp[-1 - i - 2 - prefac] * eigvec_temp[-1 - i- prefac]) > 1.e-30 ):
                exp_phi = eigvec_temp[-1 - i - 2 - prefac] / eigvec_temp[-1 - i- prefac] * abs(eigvec_temp[-1 - i- prefac] / eigvec_temp[-1 - i -2- prefac])
           
                theta = np.arctan(abs(eigvec_temp[-1 - i- prefac] / eigvec_temp[-1 - i -2- prefac]))
           
            D_phi = np.identity(4,dtype=np.complex_)
            if prefac == 0:
                D_phi [2,2] = 1/exp_phi
                D_phi [3,3]= exp_phi
            else:
                D_phi [2,2] = exp_phi
                D_phi [3,3]= 1/exp_phi


            G_theta = np.identity(4) * np.cos(theta)
            
            for j in range (2):
                G_theta[0 + j,2 + j] = np.sin(theta) 
                G_theta[2 + j,0 + j] = -np.sin(theta) 

            U_sub_new = np.bmat([[np.identity(sub_dim - 4-i), np.zeros((sub_dim - 4-i,4 + i))],[np.zeros((4,sub_dim - 4 - i)), G_theta @ D_phi, np.zeros((4, i))],[np.zeros((i, sub_dim -i)),np.identity(i)]])
            eigvec_temp = np.einsum('ij,j->i',U_sub_new,eigvec_temp)
            print(eigvec_temp)
            U_sub = U_sub_new @ U_sub

    #print(np.einsum('ij,j->i',U_sub,eigvec))
    last_gate_type = 1# 1:Givens, 2:Bog
    if abs(eigvec_temp[0]) > abs(eigvec_temp[1]):#Givens like above
        print('last layer: Givens')
        exp_phi = 1
        theta = 0
        if abs(eigvec_temp[0] * eigvec_temp[2]) > 1.e-30:
            exp_phi = eigvec_temp[0] / eigvec_temp[2] * abs(eigvec_temp[2] / eigvec_temp[0])
    
            theta = np.arctan(abs(eigvec_temp[2] / eigvec_temp[0]))

        D_phi = np.identity(4,dtype=np.complex_)
        if prefac == 0:
                D_phi [2,2] = 1/exp_phi
                D_phi [3,3]= exp_phi
        else:
            D_phi [2,2] = exp_phi
            D_phi [3,3]= 1/exp_phi

        G_theta = np.identity(4) * np.cos(theta)

        for j in range (2):
            G_theta[0 + j,2 + j] = np.sin(theta) 
            G_theta[2 + j,0 + j] = -np.sin(theta) 

        U_sub_new = np.bmat([[ G_theta @ D_phi, np.zeros((4, sub_dim - 4))],[np.zeros((sub_dim - 4, 4)),np.identity(sub_dim - 4)]])

        eigvec_temp = np.einsum('ij,j->i',U_sub_new,eigvec_temp)

        U_sub = U_sub_new @ U_sub

    else:#Bogoliubov
        print('last layer: Bogo')
        last_gate_type = 2
        exp_phi = 1
        theta = 0
        if abs(eigvec_temp[1] * eigvec_temp[2]) > 1.e-30:
            exp_phi = eigvec_temp[1] / eigvec_temp[2] * abs(eigvec_temp[2] / eigvec_temp[1])
        
            theta = np.arctan(- abs(eigvec_temp[2] / eigvec_temp[1]))

        D_phi = np.identity(4,dtype=np.complex_)
        if prefac == 0:
            D_phi [2,2] = 1/exp_phi
            D_phi [3,3]= exp_phi
        else:
            D_phi [2,2] = exp_phi
            D_phi [3,3]= 1/exp_phi

        G_theta = np.identity(4) * np.cos(theta)

        for j in range (2):
            G_theta[j,3 - j] = -np.sin(theta) 
            G_theta[3 - j, j] = np.sin(theta) 

        U_sub_new = np.bmat([[ G_theta @ D_phi, np.zeros((4, sub_dim - 4))],[np.zeros((sub_dim - 4, 4)),np.identity(sub_dim - 4)]])
        eigvec_temp = np.einsum('ij,j->i',U_sub_new,eigvec_temp)

        U_sub = U_sub_new @ U_sub


    print(np.einsum('ij,j->i',U_sub,eigvec))
    #print('U-befire',U_sub)
        
    U_sub_full_dim = sparse.csr_matrix(np.block([[np.identity(lower), np.zeros((lower,total_dim - lower))],[np.zeros((sub_dim,lower)), U_sub, np.zeros((sub_dim, total_dim-upper))],[np.zeros((total_dim-upper,upper)),np.identity(total_dim-upper)]]))
    return U_sub_full_dim, last_gate_type
