import numpy as np
import matplotlib.pyplot as plt
def zero_to_nan(values):
    """Replace every 0 with 'nan' and return a copy."""
    return [float('nan') if x==0 else x for x in values]


filename = 'ent_entropy_Jx=0_Jy=0.7853981633974483_g=0.31_beta=0_L=300.0'
data = np.loadtxt(filename + '.txt')

num_rows, num_cols = data.shape
data_array = np.zeros((num_rows, 3))
print num_rows

for i in range (num_rows):
    data_array[i][0] = data[i,0]
    data_array[i][1] = data[i,int(data[i,0]/2)]
    data_array[i][2] = max(data[i,1:])

print data_array
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

plt.savefig('post_' + filename + '.pdf') 