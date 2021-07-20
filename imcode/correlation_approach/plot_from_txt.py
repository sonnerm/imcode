import numpy as np
import matplotlib.pyplot as plt
def zero_to_nan(values):
    """Replace every 0 with 'nan' and return a copy."""
    return [float('nan') if x==0 else x for x in values]


filename = '/Users/julianthoenniss/Documents/PhD/data/ent_entropy_Jx=0.4_Jy=0.5_g=0_beta=0_L=200'
data = np.loadtxt(filename + '.txt')

num_rows, num_cols = data.shape
data_array = np.zeros((num_rows, 3))
print (num_rows)

for i in range (num_rows):
    data_array[i][0] = data[i,0]
    data_array[i][1] = data[i,int(data[i,0]/2)]
    data_array[i][2] = max(data[i,1:])

print (data_array)
fig, ax = plt.subplots()

ax.plot( data_array[:num_rows-1,0], data_array[:num_rows-1,1],
        'ro-', label=r'$max_t S$')
ax.plot( data_array[:num_rows-1,0], data_array[:num_rows-1,2],
        'ro--', label=r'$max_t S$',color = 'green')
ax.set_xlabel(r'$t$')

ax.yaxis.set_ticks_position('both')
ax.tick_params(axis="y",direction="in")
ax.tick_params(axis="x",direction="in")
ax.legend(loc="lower right")

mac_path = '/Users/julianthoenniss/Documents/Studium/PhD/data/correlation_approach'
work_path = '/Users/julianthoenniss/Documents/PhD/data/'
fiteo1_path = '/home/thoennis/data/correlation_apporach/'
plt.savefig(filename + '_post.png')