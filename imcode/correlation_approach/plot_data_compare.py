import numpy as np
import h5py
import matplotlib.pyplot as plt
def zero_to_nan(values):
    """Replace every 0 with 'nan' and return a copy."""
    return [float('nan') if x==0 else x for x in values]

np.set_printoptions(linewidth=np.nan, precision=3, suppress=True)


filename3 = '/Users/julianthoenniss/Documents/PhD/data/ES_Corr_ent_entropy_Jx=0.019999960000000004_Jy=0.020000000000000004_g=0.0_del_t=0.2_beta=2.0_L=68'
filename2 = '/Users/julianthoenniss/Documents/PhD/data/ent_entropy_Jx=1.0_Jy=2.0_g=0.0_del_t=1_beta=1.0_L=30'
filename1 = '/Users/julianthoenniss/Documents/PhD/data/ent_entropy_Jx=1.0_Jy=2.0_g=0.0_del_t=1_beta=1.0_L=30'
filename4 = '/Users/julianthoenniss/Documents/PhD/data/ES_Corr_ent_entropy_Jx=0.15_Jy=0.09_g=0.0_del_t=2.0_beta=0.3_L=68'
filename5 = '/Users/julianthoenniss/Documents/PhD/data/ES_Corr_ent_entropy_Jx=0.2_Jy=0.12_g=0.0_del_t=2.0_beta=0.4_L=68'
filename6 = '/Users/julianthoenniss/Documents/PhD/data/ES_Corr_ent_entropy_Jx=0.5_Jy=0.3_g=0_del_t=2.0_beta=1_L=68'

filename7 = '/Users/julianthoenniss/Documents/PhD/data/ED_ent_entropy_Jx=1.0_Jy=1.0_g=0_beta=1.0_L=6'
filename8 = '/Users/julianthoenniss/Documents/PhD/data/ED_ent_entropy_Jx=1.0_Jy=1.0_g=0_beta=1.0_L=8'
filename9 = '/Users/julianthoenniss/Documents/PhD/data/ED_ent_entropy_Jx=1.0_Jy=1.0_g=0_beta=1.0_L=8'

entr_data_stored =[] 


data1 =[]# np.loadtxt(filename1 + '.txt')
data2 = np.loadtxt(filename2 + '.txt')
data3 = np.loadtxt(filename3 + '.txt')
data4 = np.loadtxt(filename4 + '.txt')
data5 = np.loadtxt(filename5 + '.txt')
data6 = np.loadtxt(filename6 + '.txt')

data7 = np.loadtxt(filename7 + '.txt')
data9 = np.loadtxt(filename9 + '.txt')
data8 = np.loadtxt(filename8 + '.txt')

with h5py.File(filename1 + '.hdf5', 'r') as f:
   entr_data = f['temp_entr']
   print(entr_data[:,:])

with h5py.File(filename1 + '.hdf5', 'r') as f:
   print(f.keys())
   data_read = f['temp_entr']
   data1 = data_read[:,:]
print('data1')
print(data1)
print('data2')
print(data2)

num_rows1, num_cols1 = data1.shape
num_rows2, num_cols2 = data2.shape
num_rows3, num_cols3 = data3.shape
num_rows4, num_cols4 = data4.shape
num_rows5, num_cols5 = data5.shape
num_rows6, num_cols6 = data6.shape


num_rows7, num_cols7 = data7.shape
num_rows8, num_cols8 = data8.shape
num_rows9, num_cols9 = data9.shape


data_array1 = np.zeros((num_rows1, 2))
data_array2 = np.zeros((num_rows2, 2))
data_array3 = np.zeros((num_rows3, 2))
data_array4 = np.zeros((num_rows4, 2))
data_array5 = np.zeros((num_rows5, 2))
data_array6 = np.zeros((num_rows6, 2))

data_array7 = np.zeros((num_rows7, 2))
data_array9 = np.zeros((num_rows9, 2))
data_array8 = np.zeros((num_rows8, 2))

for i in range (num_rows1):
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


for i in range (num_rows7):
    data_array7[i][0] = data7[i,0]
    data_array7[i][1] = data7[i,int(data7[i,0]/2)]

for i in range (num_rows8):
    data_array8[i][0] = data8[i,0]
    data_array8[i][1] = data8[i,int(data8[i,0]/2)]

for i in range (num_rows9):
    data_array9[i][0] = data9[i,0]
    data_array9[i][1] = data9[i,int(data9[i,0]/2)]



fig, ax = plt.subplots()


ax.plot( data_array1[:num_rows1-1,0], zero_to_nan(data_array1[:num_rows1-1,1]), label=r'$\Delta t = 0.05$', marker='o', alpha=.5, ms=5, color ='blue')
ax.plot( data_array2[:num_rows2-1,0], zero_to_nan(data_array2[:num_rows2-1,1]), label=r'$\Delta t = 0.1$', marker='o', alpha=.5, ms=5, color = 'red')
#ax.plot( data_array3[:num_rows3-1,0], zero_to_nan(data_array2[:num_rows3-1,1]), label=r'$\Delta t = 0.2$', marker='o', alpha=.5, ms=5)
#ax.plot( data_array4[:num_rows4-1,0], zero_to_nan(data_array4[:num_rows4-1,1]), label=r'$\Delta t = 0.3$', marker='o', alpha=.5, ms=5)
#ax.plot( data_array5[:num_rows5-1,0], zero_to_nan(data_array5[:num_rows5-1,1]), label=r'$\Delta t = 0.4$', marker='o', alpha=.5, ms=5)
#ax.plot( data_array6[:num_rows6-1,0], zero_to_nan(data_array6[:num_rows6-1,1]), label=r'$\Delta t = 1.0$', marker='o', alpha=.5, ms=5)

#ax.plot(np.arange(30),30*[1.0])

#ax.plot( data_array7[:num_rows7-1,0], zero_to_nan(data_array7[:num_rows7-1,1]),
#        '--', label='ED_'+r'$L_{E}=6$')

#ax.plot( data_array8[:num_rows8-1,0], zero_to_nan(data_array8[:num_rows8-1,1]),
#        'o', label='ED_'+r'$L_{E}=8$')
#ax.plot( data_array9[:num_rows9-1,0], zero_to_nan(data_array9[:num_rows9-1,1]),
#        'x', label='ED_'+r'$L_{E}=10$')


ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$S$')
#ax.set_xscale('log')
#ax.set_ylim([-1.e-13,1.e-13])

ax.yaxis.set_ticks_position('both')
ax.tick_params(axis="y",direction="in")
ax.tick_params(axis="x",direction="in")
ax.legend(loc="upper left")

mac_path = '/Users/julianthoenniss/Documents/Studium/PhD/data/correlation_approach'
work_path = '/Users/julianthoenniss/Documents/PhD/data/'
fiteo1_path = '/home/thoennis/data/correlation_apporach/'
plt.savefig(work_path + 'Log_FS_Corr_ent_entropy_Jx=0.5_Jy=0.5_g=0' + '_L=68_inset.pdf')