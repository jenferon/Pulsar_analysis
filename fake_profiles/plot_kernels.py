import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

x=np.linspace(-2,2,1000)
h = np.abs(x[2]-x[3])
#triangle 
y_tri = []
bin_width = 0.5
for i in range(0,len(x)):
	if np.abs(x[i]) <= bin_width:
		y_tri = np.append(y_tri,1-np.abs(x[i])/bin_width)
	else:
		y_tri = np.append(y_tri,0)

#gaussian
bin_width = 0.25
y_gauss = (1/np.sqrt(2*np.pi*bin_width))*np.exp(-0.5*(x/bin_width)**2)

#rectangle 
y_rec = []
bin_width = 0.5
for i in range(0,len(x)):
	if np.abs(x[i]) <= bin_width:
		y_rec = np.append(y_rec,1)
	else:
		y_rec = np.append(y_rec,0)

#sinc 
bin_width = 1
y_sinc = []
for i in range(0,len(x)):
	if x[i] == 0:
		y_sinc = np.append(y_sinc,1)
	else:
		y_sinc = np.append(y_sinc, np.abs(np.sinc(np.pi*x[i]*bin_width)))


plt.plot(x,y_tri, label="triangle (h=0.5)")
plt.plot(x,y_sinc, label="sinc (h=1)")
plt.plot(x,y_rec, label="rectangle (h=0.5)")
plt.plot(x, y_gauss, label="gaussian (h=0.25)")
plt.legend()
plt.ylabel("Weighting")
plt.xlabel("Difference between observation times: $t_x - t_y - lags$")
plt.savefig("Kernels_for_CCF.png")
plt.show()
