import numpy as np
import matplotlib.pyplot as plt

cutoff=58000

data = np.loadtxt('widths_formatted.txt')

max_diff = np.abs(data[:,0]-cutoff)
cutoff_arg = np.argmin(max_diff)

data = data[:cutoff_arg,:]

np.savetxt('cut_widths_formatted.dat', data)

plt.errorbar(data[:,0],data[:,1], yerr = data[:,2])
plt.savefig('width_plotted.png')
plt.show()
