from numpy import dtype
from numpy.matrixlib import bmat
from scipy import linalg
import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys
from scipy.linalg.decomp import eig
from scipy.linalg import det
from scipy.sparse.linalg import eigsh
import scipy
# generate random floating point values
from random import seed
from random import random
from compute_generators import compute_generators
from scipy import linalg


np.set_printoptions(threshold=sys.maxsize, precision=6, suppress=True)
np.set_printoptions(linewidth=470)

def create_Magnus_Floquet_ham(Jx, Jy, g,L,mu=0):#creates first order Magnus Hamiltonan WITHOUT chemical potential
    # Floquet Hamiltonian of xy-Model:
    Jp = Jx + Jy
    Jm = Jy - Jx
    Delta = Jp**2 - Jm**2
    alpha_XY = 0.5j* Delta 
    alpha_Ising = 1.j* Jm * g

    H = np.zeros((2*L, 2*L), dtype=np.complex_)
    
    for i in range(L-1):
        #sum of H_even and H_odd
        H[i, i+1] += Jp/2
        H[i+L+1, i+L] += -Jp/2

        H[i, i+L+1] += -Jm/2
        H[i+1, i+L] += Jm/2

        H[i,i] += - (g + mu/2) 
        H[i+L,i+L] += (g + mu/2) 

        #commutator (first order in del_t)
        
        #for KIC case
        H[i, i+L+1] += alpha_Ising 
        H[i+1, i+L] += - alpha_Ising 

        #for XY case
        if i < (L - 2):
            H[i,i+2] += alpha_XY/2 * (-1)**i
            H[i+L+2,i+L] += -alpha_XY/2 * (-1)**i
        
    #add last term that is not contained in above for loop
    H[L-1, L-1] += - (g + mu/2) 
    H[2*L - 1, 2*L - 1] += (g + mu/2) 

    mag = 1.e-8
    #(anti-) periodic boundary conditions (last factor switches between periodc and antiperiodic boundary conditions depending on length of chain)
    H[L-1,0] += mag* (-1)**(L+1)
    H[L,2*L-1] +=- mag* (-1)**(L+1)

    H[L-1,L] += - mag* (-1)**(L+1)
    H[0,2*L-1] += mag* (-1)**(L+1)
    
    mag = 1.e-6
    H += H.T.conj() #add hermitian conjugate
    seed(1)
    stabilizer = np.zeros((2*L,2*L))
    for i in range(L):
        stabilizer[i,i] = (random()-0.5) * mag
        stabilizer[i+L,i+L] = - stabilizer[i,i]
    H += stabilizer
    #print('H')
    #print(H)
    
    return H

def create_exact_Floquet_ham(Jx, Jy, g, L, mu = 0):#creates exact Hamiltonan WITHOUT chemical potential
   
    G_even, G_odd, G_kick,G1 = compute_generators(L,Jx,Jy,g,0)

    #H =  -1.j * linalg.logm(linalg.expm(.5j * G_kick) @ linalg.expm(.5j * G_even)  @ linalg.expm(.5j * G_odd))
    H =  -.5j * linalg.logm( linalg.expm(1.j * G_kick) @ linalg.expm(1.j * G_even) @ linalg.expm(1.j * G_odd) )
    return H


def compute_BCS_Kernel(Jx, Jy, g, mu, L, filename):

#check for right ordering of eigenvectors in matrix M
    """
    eigvals_ord, eigvecs_ord= linalg.eigh(create_Floquet_ham(Jx,Jx,g,L))
    #print('eigvecs_ord')
    #print(eigvecs_ord)
    ordering = []
    for i in range(2*L):
        if eigvecs_ord[0,i] != 0:
            ordering.append(i)
    #print('ordering')
    #print(ordering)
    """
    np.set_printoptions(threshold=sys.maxsize, precision=10, suppress=True)
    #create Floquet Hamiltonian of interest
    #exact effective Hamiltonian
    H_eff = create_exact_Floquet_ham(Jx,Jy,g,L)
    #print('H_eff')
    #print(H_eff)

    #Magnus expansion
    #H_eff_mag = create_Magnus_Floquet_ham(Jx,Jy,g,L,mu)
    #print('H_eff')
    #print(H_eff-H_eff_mag)
    

    eigvals, eigvecs = linalg.eigh(H_eff)
    #print('eigenvalues of eff. Hamiltonian:',eigvals)
    """
    M = np.zeros((2*L,2*L), dtype=np.complex_)
    for i in range(L):  # sort eigenvectors and eigenvalues such that the first half are the ones with positive real part, and the second half have negative real parts
        M[:, i] = eigvecs[:, ordering[i]]
        M[0:L,i+L] = M[L:2*L, i].conj()
        M[L:2*L,i+L] = M[0:L, i].conj()
    """
    argsort = np.argsort(- np.real(eigvals))
    M = np.zeros((eigvecs.shape), dtype=np.complex_)
   
    for i in range(L):  # sort eigenvectors and eigenvalues such that the first half are the ones with positive real part, and the second half have negative real parts
        M[:, i] = eigvecs[:, argsort[i]]
        M[:, 2 * L - 1 - i] = eigvecs[:, argsort[i + L]]

    #print(M)
    #check that matrix is diagonalized by M
    #diag = M.T.conj() @ H_eff @ M
    #print(diag)
    #print('Diagonal')


    U = M[0:L,0:L]
    V = M[L:2*L,0:L]

    #print('u,v')
    #print(U)
    #print(V)
    Z = - linalg.inv(U.T.conj()) @ V.T.conj()

    Corr = np.bmat([[U@U.T.conj(),U@V.T.conj()],[V@U.T.conj(),V@V.T.conj()]])
    print('eig(Corr)')
    #print(linalg.eigvalsh(Corr))
    antisym_check = 0
    sum = 0
    for i in range (len(Z[0])):
        for j in range (len(Z[0])):
        
            antisym_check += abs(Z[i,j] + Z[j,i]) 
            sum +=  abs(Z[i,j]) + abs(Z[j,i])

    antisym_check *= 1/sum
    print('antisym_check')
    print(antisym_check)

    #print('Z')
    #print(Z)
    
    #print(linalg.eigvals(Z))

    with h5py.File(filename + ".hdf5", 'a') as f:
        init_BCS_data = f['init_BCS_state']
        init_BCS_data[:,:] = Z[:,:]
    
    DM_compact = 0.5* np.bmat([[Z.T.conj(),np.zeros((L,L))],[np.zeros((L,L)),Z]])
    #print('DM_compact')
    #print(DM_compact)
    U =  4 * U #this is just a convenient, qn artifially introduced renormalization so numbers dont become too large..it is precisely cancelled by factor in determinant when IM-matrix element is computed. Real norm of this state is without this factor
    normalization = 1#1 / abs(det(U)) #this normalizes the DM (the state has the squareroot of this as normalization) (the value 1 is just set here because in the new code, the normalization is not used and it leads to problems when Jx = Jy)

    return 2*DM_compact, normalization
