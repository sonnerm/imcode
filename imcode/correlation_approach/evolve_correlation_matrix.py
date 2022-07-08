import numpy as np
from numpy import dtype, version
from numpy.lib.type_check import imag
from scipy.linalg import expm, schur, eigvals
from scipy import linalg
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from numpy.linalg import multi_dot
import scipy.optimize
import h5py
from compute_generators import compute_generators


nsites = 201
Jx=.1
Jy=.1
g=.0
beta_tilde=beta_tilde = 0.5 * np.log( (1 + np.tan(Jx) * np.tan(Jy) ) / (1 - np.tan(Jx) * np.tan(Jy) ) + 0.j) 

G_XY_odd = np.zeros((2 * nsites, 2 * nsites))
G_XY_even = np.zeros((2 * nsites, 2 * nsites))

Jp = (Jx + Jy)
Jm = (Jy - Jx)

if abs(Jm) < 1e-10:
    Jm = 1e-10
if abs(g) < 1e-10:
    g = 1e-10

eps = 0#1e-8  # lift degeneracy
G_XY_odd[0, nsites - 1] += eps
G_XY_odd[nsites - 1, 0] += eps
G_XY_odd[nsites, 2 * nsites - 1] += -eps
G_XY_odd[2 * nsites - 1, nsites] += -eps

G_XY_odd[nsites - 1, nsites] -= eps
G_XY_odd[0, 2 * nsites - 1] += eps
G_XY_odd[2 * nsites - 1, 0] += eps
G_XY_odd[nsites, nsites - 1] -= eps

for i in range(1, nsites - 1, 2):
    G_XY_even[i, i + 1] = Jp
    G_XY_even[i + 1, i] = Jp
    G_XY_even[i, i + nsites + 1] = -Jm
    G_XY_even[i + 1, i + nsites] = Jm
    G_XY_even[i + nsites, i + 1] = Jm
    G_XY_even[i + nsites + 1, i] = -Jm
    G_XY_even[i + nsites, i + nsites + 1] = -Jp
    G_XY_even[i + nsites + 1, i + nsites] = -Jp

for i in range(0, nsites - 1, 2):
    G_XY_odd[i, i + 1] = Jp
    G_XY_odd[i + 1, i] = Jp
    G_XY_odd[i, i + nsites + 1] = -Jm
    G_XY_odd[i + 1, i + nsites] = Jm
    G_XY_odd[i + nsites, i + 1] = Jm
    G_XY_odd[i + nsites + 1, i] = -Jm
    G_XY_odd[i + nsites, i + nsites + 1] = - Jp
    G_XY_odd[i + nsites + 1, i + nsites] = - Jp

# G_g - single body kicks
G_g = np.zeros((2 * nsites, 2 * nsites))
for i in range(nsites):
    G_g[i, i] = - 2 * g
    G_g[i + nsites, i + nsites] = 2 * g


U = expm(1.j*G_g) @ expm(1.j * G_XY_even) @ expm(1.j * G_XY_odd)

print(U[:5,:5])

Lambda = np.zeros((2*nsites,2*nsites))
with h5py.File('/Users/julianthoenniss/Documents/PhD/data/corr_mich/Jx=0.3_Jy=0.3_g=0_L=200_FermiSea_correlations' + '.hdf5', 'r') as f:
    print(f.keys())
    entr_data = f['corr_realspace=']

    Lambda[1:nsites,1:nsites] = entr_data[:nsites-1,:nsites-1]
    Lambda[nsites+1:,1:nsites] = entr_data[nsites-1:,:nsites-1]
    Lambda[1:nsites,nsites+1:] = entr_data[:nsites-1,nsites-1:]
    Lambda[nsites+1:,nsites+1:] = entr_data[nsites-1:,nsites-1:]
    #print(entr_data[:nsites-1,:nsites-1].shape)
    #print(entr_data[nsites-1:,:nsites-1].shape)
    #print(entr_data[:nsites-1,nsites-1:].shape)
    #print(entr_data[nsites-1:,nsites-1:].shape)
    #Lambda = entr_data[:,:]
    

Lambda[0,0] = 1
Lambda[nsites, nsites] = 0

zz = []

Lambda = np.zeros((2*nsites,2*nsites))
Lambda[0,0] = 1
Lambda[nsites, nsites] = 0
for i in range (1,nsites):
    Lambda[i,i] = 0
    Lambda[i+nsites,i+nsites] = 1

for i in range (200):
    Lambda = U.T @ Lambda @ U.conj()
    zz.append(Lambda[0,0])





def cl(va,lt,cmap='Set2',invert=False,margin=0.1,lowcut=0,upcut=1): 
    cmap = plt.cm.get_cmap(cmap)
    ind=list(lt).index(va) 
    if len(lt)>1:
        rt=ind/(len(lt)-1)
        rt*=1-min(1-margin,lowcut+1-upcut)
        rt+=lowcut
    else:
        rt=0
        rt+=lowcut
    rt=rt*(1-2*margin)+margin
    if invert:
        rt=1-rt    
    return cmap(rt)



plt.rcParams.update({
  "text.usetex": True,
  "font.family": "Helvetica",
    "font.size" : 8
})
fig, ax = plt.subplots()
fig.set_size_inches(3.5,2.2) 
z=0
ax.plot( np.arange(len(zz)), np.real(np.array(zz)), label=r'$\langle c^\dagger_0(t) c_0(t) \rangle_{Vac.}$',linewidth=.8, ms=.5,zorder=-1,color=cl(z,[0,1,2,3,4,5,6,7]))

zz_inf = []
Lambda = 0.5 * np.identity(2*nsites)
Lambda[0,0] = 1
Lambda[nsites, nsites] = 0

for i in range (200):
    Lambda = U.T @ Lambda @ U.conj()
    zz_inf.append((Lambda[0,0]))

z=1
ax.plot( np.arange(len(zz_inf)), np.real(np.array(zz_inf)-0.5) * zz[1]/(zz_inf[1]-0.5),label=r'$2(\langle c^\dagger_0(t) c_0(t) \rangle_{Inf.Temp.}-0.5)$', linewidth=.8,alpha=0.8,linestyle = '--', ms=1,zorder=-1,color=cl(z,[0,1,2,3,4,5,6,7]))
z=2
ax.plot( np.arange(1,len(zz)),40*np.arange(1,len(zz), dtype=float)**-3.,label=r'$t^{-3}$',linestyle ='--', linewidth=.8,alpha=0.8, ms=1,zorder=-1,color='black')

z=3
#ax.plot( np.arange(1,len(zz)),np.array((len(zz)-1)*[0.5]),label=r'$const.=0.5$',linestyle =':', linewidth=.8,alpha=0.8, ms=1,zorder=-1,color='black')


ax.set_xlabel(r'$t$')
ax.set_yscale('log')
ax.set_xscale('log')
#ax.set_ylim([.0001,1])
#ax.set_ylabel(r'$\langle c^\dagger_0(t) c_0(t) \rangle$')

ax.legend()


#plt.show()
plt.savefig('/Users/julianthoenniss/Documents/PhD/data/exact_evolution_rescaled.pdf', bbox_inches='tight')