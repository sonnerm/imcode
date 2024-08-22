#!/usr/bin/env python
# coding: utf-8

# # Calculating an observable of an interacting impurity in a fermionic bath
# ## Preparations
# First make sure that you have installed the latest versions of imcode, freeferm and ttarray.
# 
# ```
# git clone https://github.com/sonnerm/imcode
# pip install --editable ./imcode/
# git clone https://github.com/sonnerm/freeferm
# pip install --editable ./freeferm/
# git clone https://github.com/sonnerm/ttarray
# pip install --editable ./ttarray/
# ```
# It is recommended to use a virtual environment since those libraries are under active development. You'll also need numpy, scipy, h5py and matplotlib. 
# 

# In[2]:


import imcode
import ttarray
import freeferm
import numpy as np
import scipy.linalg as sla
import h5py
import matplotlib.pyplot as plt


# ## Define spectral density
# We use the spectral density found in the paper by Cohen et. al. $f(\omega)=\frac{\Gamma}{(1+e^{\nu(x-\epsilon_c)})(1+e^{-\nu(x+\epsilon_c)})}$

# In[3]:


def spec_dens(x):
    e_c = 10.
    nu = 10.
    return  2 /((1+np.exp(nu*(x - e_c))) * (1+np.exp(-nu*(x + e_c))))


# ## Compute the Influence matrix as Gaussian state
# We now represent the IM as $|I\rangle \propto e^{i c^\dagger_i B_{ij} c^\dagger_j}|\Omega\rangle$

# In[9]:


omin=-12 #integration bounds
omax=12
beta=50.0 #inverse temperature
mu=0.0 #chemical potential
tmax=2.0
nsteps=40 #just for computing things fast on the laptop
nsubsteps=2 #number of subdivisions
bmat=imcode.spectral_density_to_fermiexp(spec_dens,omin,omax,beta,mu,tmax,nsteps,nsubsteps)


# ## Convert to correlation matrix
# My Fishmann-White implementation needs the correlation matrix in majorana form, that is
# $$\Lambda_{ij} = \langle \gamma_i \gamma_j \rangle$$

# In[10]:


bcorr=imcode.fermiexp_to_fermicorr(bmat)


# ## Convert Majorana correlation matrix to circuit
# This routine uses the Majorana Fishman-White algorithm to obtain the circuit which rotates the vacuum to the Gaussian state described by our correlation matrix. For the purpose of this demonstration, choosing a cutoff of $10^{-8}$ is definitely sufficient.

# In[11]:


circuit=imcode.fermicorr_to_circuit(bcorr,nbcutoff=1e-8)


# ## Convert circuit to MPS
# To obtain the Influence Matrix MPS we apply the circuit to the vacuum MPS. For faster computations, we choose a maximal bond dimension $\chi=128$ and a (relative) svd-cutoff of $10^{-12}$.

# In[17]:


mps_fermi=imcode.circuit_to_mps(circuit,nsteps,chi=128,svdcutoff=1e-12)


# ## Change the Jordan Wigner order
# The Fishmann-White IM is quantized according to the MPS-leg order. However, if we want our impurity to be standard gates one would use in ED, it is necessary to change the Jordan Wigner order. The resulting MPS should be exactly what one would get from contracting a chain mapping in the transverse direction.

# In[18]:


mps_spin=imcode.fermi_to_spin(mps_fermi)


# ## Define impurity initial state and channel
# For this demonstration we want to compute the Anderson Impurity model to compare to Cohen et.al. In spin basis the unitary time evolution operator is given by:
# $$\hat{U} = e^{i\mathrm{d}t U/4 (S^Z \otimes S^z)}$$
# To break the spin symmetry our initial state will be spin polarized; the spin up state is occupied, the spin down state is empty.

# In[19]:


SZ=np.diag([1,-1])
ID=np.eye(2)
SX=np.array([[0,1],[1,0]])
SY=np.array([[0,-1j],[1j,0]])
dt = .1
U = 8.
channel=imcode.unitary_channel(sla.expm(-dt*1j*U/4*(np.kron(SZ,SZ))))
init=np.kron(np.diag([1,0]),np.diag([0,1])) #spin up occupied, spin down unoccupied


# ## Compute the density matrix time evolution
# We now compute the density matrix time evolution of the impurity coupled to two spin species.

# In[20]:


dms=list(imcode.brickwork_embedded_evolution(mps_spin,channel,mps_spin,init,normalize=True))


# ## Plot results
# 
# We plot observables only from every second density matrix since they correspond to before/after the impurity action.
# The results are compared with data points from Cohen et. al.
# 
# All done 🙂

# In[30]:


import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0
font = {'family' : 'Sans',
    'weight' : 'normal'}
matplotlib.rc('font', **font)
matplotlib.rcParams['text.usetex'] =True
matplotlib.rc('font', family='Helvetica',size=18)


plt.plot(np.linspace(0,2.0,41,endpoint=True),[r[0,0]/np.trace(r) for r in dms[::2]],label=r"$|\uparrow\downarrow\rangle$")
plt.plot(np.linspace(0,2.0,41,endpoint=True),[r[1,1]/np.trace(r) for r in dms[::2]],label=r"$|\uparrow\rangle$")
plt.plot(np.linspace(0,2.0,41,endpoint=True),[r[2,2]/np.trace(r) for r in dms[::2]],label=r"$|\downarrow\rangle$")
plt.plot(np.linspace(0,2.0,41,endpoint=True),[r[3,3]/np.trace(r) for r in dms[::2]],label=r"$|0\rangle$")
plt.legend()
plt.xlabel('t')
plt.ylabel(r'$\rho_{ii}$')


# In[ ]:




