import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
plt.rcParams.update({
  "text.usetex": True,
  "font.family": "Helvetica"
})



def zero_to_nan(values):
    """Replace every 0 with 'nan' and return a copy."""
    return [float('nan') if x==0 else x for x in values]

np.set_printoptions(linewidth=np.nan, precision=3, suppress=True)


filename1 = '/Users/julianthoenniss/Documents/PhD/data/prod_thermal_Jx=0.3_Jy=0.5_g=0.0mu=0_del_t=1.0_beta=0.0_L=400'
filename2 = '/Users/julianthoenniss/Documents/PhD/data/prod_thermal_Jx=0.3_Jy=0.5_g=0.0mu=0_del_t=1.0_beta=0.4_L=400'
filename3 = '/Users/julianthoenniss/Documents/PhD/data/prod_thermal_Jx=0.3_Jy=0.5_g=0.0mu=0_del_t=1.0_beta=1.0_L=400'
filename4 = '/Users/julianthoenniss/Documents/PhD/data/prod_thermal_Jx=0.3_Jy=0.5_g=0.0mu=0_del_t=1.0_beta=2.0_L=400'
filename5 = '/Users/julianthoenniss/Documents/PhD/data/prod_thermal_Jx=0.3_Jy=0.5_g=0.0mu=0_del_t=1.0_beta=5.0_L=400'
filename6 = '/Users/julianthoenniss/Documents/PhD/data/prod_thermal_Jx=0.3_Jy=0.5_g=0.0mu=0_del_t=1.0_beta=10.0_L=400'
entr_data_stored =[] 


data1 =[]# np.loadtxt(filename1 + '.txt')
data2 = []
data3 = []
data4 = []
data5 = []
data6 = []

with h5py.File(filename1 + '.hdf5', 'r') as f:
   data_read = f['temp_entr']
   data1 = data_read[:,:]
with h5py.File(filename2 + '.hdf5', 'r') as f:
   data_read = f['temp_entr']
   data2 = data_read[:,:]

with h5py.File(filename3 + '.hdf5', 'r') as f:
   data_read = f['temp_entr']
   data3 = data_read[:,:]
with h5py.File(filename4 + '.hdf5', 'r') as f:
   data_read = f['temp_entr']
   data4 = data_read[:,:]

with h5py.File(filename5 + '.hdf5', 'r') as f:
   data_read = f['temp_entr']
   data5 = data_read[:,:]
with h5py.File(filename6 + '.hdf5', 'r') as f:
   data_read = f['temp_entr']
   data6 = data_read[:,:]
#times1 = np.concatenate((np.array(data1[:10,0]),np.arange(14,70,4)), axis=None)
#times2 = np.concatenate((np.array(data2[:10,0]),np.arange(14,70,4)), axis=None)
#times3 = np.concatenate((np.array(data3[:10,0]),np.arange(14,70,4)), axis=None)
#times4 = np.concatenate((np.array(data4[:10,0]),np.arange(14,70,4)), axis=None)

#print(times1)
num_rows1, num_cols1 = data1.shape
num_rows2, num_cols2 = data2.shape
num_rows3, num_cols3 = data3.shape
num_rows4, num_cols4 = data4.shape
num_rows5, num_cols5 = data5.shape
num_rows6, num_cols6 = data6.shape


data_array1 = np.zeros((num_rows1, 2))
data_array2 = np.zeros((num_rows2, 2))
data_array3 = np.zeros((num_rows3, 2))
data_array4 = np.zeros((num_rows4, 2))
data_array5 = np.zeros((num_rows5, 2))
data_array6 = np.zeros((num_rows6, 2))

for i in range (num_rows1-13):
    data_array1[i][0] = data1[i,0] 
    data_array1[i][1] = np.amax(data1[i,1:]) #data1[i,int(data1[i,0]/2)]

for i in range (num_rows2):
    data_array2[i][0] = data2[i,0]
    data_array2[i][1] = np.amax(data2[i,1:])#data2[i,int(data2[i,0]/2)] 

for i in range (num_rows3):
    data_array3[i][0] = data3[i,0]
    data_array3[i][1] = np.amax(data3[i,1:])#data3[i,int(data3[i,0]/2)]

for i in range (num_rows4):
    data_array4[i][0] = data4[i,0] 
    data_array4[i][1] = np.amax(data4[i,1:])

for i in range (num_rows5):
    data_array5[i][0] = data5[i,0]
    data_array5[i][1] = np.amax(data5[i,1:])

for i in range (num_rows6):
    data_array6[i][0] = data6[i,0]
    data_array6[i][1] = np.amax(data6[i,1:])

fig, ax = plt.subplots()

name = "tab10"
cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
colors = cmap.colors  # type: list
ax.set_prop_cycle(color=colors)

ax.plot( data_array1[:num_rows1-1,0], zero_to_nan(data_array1[:num_rows1-1,1]), label=r'$\beta = 0.0$',linestyle='-',marker='o',alpha=1, ms=2)
ax.plot( data_array2[:num_rows2-1,0], zero_to_nan(data_array2[:num_rows2-1,1]), label=r'$\beta = 0.4$',linestyle='-', marker='o', alpha=1, ms=2)
ax.plot( data_array3[:num_rows3-1,0], zero_to_nan(data_array3[:num_rows3-1,1]), label=r'$\beta = 1.0$',linestyle='-', marker='o', alpha=1, ms=2)
ax.plot( data_array4[:num_rows4-1,0], zero_to_nan(data_array4[:num_rows4-1,1]), label=r'$\beta = 2.0$',linestyle='-', marker='o', alpha=1, ms=2)
ax.plot( data_array5[:num_rows5-1,0], zero_to_nan(data_array5[:num_rows5-1,1]), label=r'$\beta = 5.0$',linestyle='-', marker='o', alpha=1, ms=2)
ax.plot( data_array6[:num_rows6-1,0], zero_to_nan(data_array6[:num_rows6-1,1]), label=r'$\beta = 10.0$', marker='x', linestyle=':',alpha=1, ms=2, color='black')

#ax.plot(np.arange(30),30*[1.0])

#ax.plot( data_array7[:num_rows7-1,0], zero_to_nan(data_array7[:num_rows7-1,1]),
#        '--', label='ED_'+r'$L_{E}=6$')

#ax.plot( data_array8[:num_rows8-1,0], zero_to_nan(data_array8[:num_rows8-1,1]),
#        'o', label='ED_'+r'$L_{E}=8$')
#ax.plot( data_array9[:num_rows9-1,0], zero_to_nan(data_array9[:num_rows9-1,1]),
#        'x', label='ED_'+r'$L_{E}=10$')


ax.set_xlabel('Floquet Time Steps ')
#ax.set_ylabel(r'$S_{max}(t)$')
#ax.set_xlabel(r'$t$' , labelpad=0)
ax.set_ylabel('Max. Temp. Entanglement Entropy ')
#ax.set_xscale('log')
ax.set_xlim(left=0, right=300)
ax.set_ylim(bottom=0.5, top=1.7)
ax.xaxis.set_major_locator(MultipleLocator(50))
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_minor_locator(MultipleLocator(0.05))
ax.xaxis.set_minor_locator(MultipleLocator(10))


#Move ticks into plot area
ax.tick_params(axis='x', which='minor', direction="in")
ax.tick_params(axis='x', which='major', direction="in")
ax.tick_params(axis='y', which='minor', direction="in")
ax.tick_params(axis='y', which='major', direction="in")


#Inset
ax2 = plt.axes([0,0,1,1])
ax2.set_xscale('log')
ax2.xaxis.set_major_locator(MultipleLocator(100))
ax2.yaxis.set_major_locator(MultipleLocator(0.1))
ax2.yaxis.set_minor_locator(MultipleLocator(0.05))
ax2.xaxis.set_minor_locator(MultipleLocator(20))
ax2.xaxis.set_minor_formatter(plt.NullFormatter())

ax2.tick_params(axis='x', which='minor', direction="in")
ax2.tick_params(axis='x', which='major', direction="in")
ax2.tick_params(axis='y', which='minor', direction="in")
ax2.tick_params(axis='y', which='major', direction="in")
# Manually set the position and relative size of the inset axes within ax1
ip = InsetPosition(ax, [0.14,0.08,0.58,0.55])
ax2.set_axes_locator(ip)
# Mark the region corresponding to the inset axes on ax1 and draw lines
# in grey linking the two axes.
x1, x2, y1, y2 = 100,300,1.35,1.7
ax2.set_xlim(x1, x2)
ax2.set_ylim(y1, y2)
ret = mark_inset(ax, ax2, loc1=2, loc2=4, fc="none", ec='0.0')

#to make connector line appear BEHIND legend
for bc in ret[1:]:
    bc.remove()
    ax.add_patch(bc)
    bc.set_zorder(4)
    bc.set_clip_on(True)

ax2.plot( data_array1[:num_rows1-1,0], zero_to_nan(data_array1[:num_rows1-1,1]), label=r'$\beta = 0.0$',linestyle='-',marker='o',alpha=1, ms=2)
ax2.plot( data_array2[:num_rows2-1,0], zero_to_nan(data_array2[:num_rows2-1,1]), label=r'$\beta = 0.4$',linestyle='-', marker='o', alpha=1, ms=2)
ax2.plot( data_array3[:num_rows3-1,0], zero_to_nan(data_array3[:num_rows3-1,1]), label=r'$\beta = 1.0$',linestyle='-', marker='o', alpha=1, ms=2)
ax2.plot( data_array4[:num_rows4-1,0], zero_to_nan(data_array4[:num_rows4-1,1]), label=r'$\beta = 2.0$',linestyle='-', marker='o', alpha=1, ms=2)
ax2.plot( data_array5[:num_rows5-1,0], zero_to_nan(data_array5[:num_rows5-1,1]), label=r'$\beta = 5.0$',linestyle='-', marker='o', alpha=1, ms=2)
ax2.plot( data_array6[:num_rows6-1,0], zero_to_nan(data_array6[:num_rows6-1,1]), label=r'$\beta = 10.0$', marker='x', linestyle=':',alpha=1, ms=2, color='black')

ax.legend(loc='lower right', fancybox=False, shadow=False, edgecolor='black').set_zorder(1000)

# set the linewidth of each legend object
# set the linewidth of each legend object


mac_path = '/Users/julianthoenniss/Documents/Studium/PhD/data/correlation_approach'
work_path = '/Users/julianthoenniss/Documents/PhD/data/'
fiteo1_path = '/home/thoennis/data/correlation_apporach/'
plt.savefig(work_path + 'Prod_Corr_ent_entropy_Jx=0.3_Jy=0.5_g=0' + '_L=400.pdf')