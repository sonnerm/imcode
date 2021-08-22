import numpy as np
import matplotlib.pyplot as plt
def zero_to_nan(values):
    """Replace every 0 with 'nan' and return a copy."""
    return [float('nan') if x==0 else x for x in values]


filename1 = '/Users/julianthoenniss/Documents/PhD/data/Corr_ent_entropy_Jx=0_Jy=0.7_g=0.2_L=8'
filename2 = '/Users/julianthoenniss/Documents/PhD/data/ED_ent_entropy_Jx=0_Jy=0.7_g=0.2_L=8'
data1 = np.loadtxt(filename1 + '.txt')
data2 = np.loadtxt(filename2 + '.txt')


num_rows1, num_cols1 = data1.shape
num_rows2, num_cols2 = data2.shape

data_array1 = np.zeros((num_rows1, 2))
data_array2 = np.zeros((num_rows2, 2))

for i in range (num_rows1):
    data_array1[i][0] = data1[i,0]
    data_array1[i][1] = data1[i,int(data1[i,0]/2)]

for i in range (num_rows2):
    data_array2[i][0] = data2[i,0]
    data_array2[i][1] = data2[i,int(data2[i,0]/2)]

fig, ax = plt.subplots()

ax.plot( data_array1[:num_rows1-1,0], zero_to_nan(data_array1[:num_rows1-1,1]),
        'ro-', label='Correlation appr.')
ax.plot( data_array2[:num_rows2-1,0], zero_to_nan(data_array2[:num_rows2-1,1]),
        '.--', label='ED',color = 'green')
ax.set_xlabel(r'$t$')

ax.yaxis.set_ticks_position('both')
ax.tick_params(axis="y",direction="in")
ax.tick_params(axis="x",direction="in")
ax.legend(loc="lower right")

mac_path = '/Users/julianthoenniss/Documents/Studium/PhD/data/correlation_approach'
work_path = '/Users/julianthoenniss/Documents/PhD/data/'
fiteo1_path = '/home/thoennis/data/correlation_apporach/'
plt.savefig(work_path + 'Comp_Jx=0_Jy=0.7_g=0.2_L=8' + '_post.png')