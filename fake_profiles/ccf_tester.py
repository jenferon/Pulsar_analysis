import numpy as np
import matplotlib.pyplot as plt
from ccf_interpolate import ccf

t_min = 0
t_max = 30
t_spacing = 0.1

t_x = np.arange(t_min, t_max, t_spacing)
t_y = np.linspace(t_min, t_max, t_spacing)

x = np.sin(np.pi*t_x)
y = np.sin(t_y)

max_lag = np.pi
min_lag = -np.pi
lag_nos = np.arange(-100, 100, 1)
lags = lag_nos*t_spacing

ccf_test = ccf(x, x, t_x, t_x, method = "gaussian", lags=lags, min_weighting = 0,  max_gap=100).results()


plt.errorbar(ccf_test[:,0], ccf_test[:,1], ccf_test[:,2])
plt.show()

