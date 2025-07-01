import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('width_mjd.txt')
checked_data = np.array((0,0,0))
for i in range(len(data[:,0])):
	if data[i,1]>data[i,2]:
		checked_data = np.vstack((checked_data, data[i,:]))
	else:
		print(data[i,0])
checked_data = np.delete(checked_data, 0,0)

plt.errorbar(checked_data[:,0],checked_data[:,1], yerr = checked_data[:,2])
plt.savefig('width_plotted.png')
plt.show()
