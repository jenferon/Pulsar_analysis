import numpy as np
import matplotlib.pyplot as plt 

data = np.loadtxt("DFB_edited_freq.dat")
mjd = data[:,1]
snr = data[:,0]
#print(mjd)
plt.plot(mjd,snr)
plt.xlabel("MJD")
plt.ylabel("Signal/Noise")
plt.savefig("snr_DFB.png")
plt.show()
