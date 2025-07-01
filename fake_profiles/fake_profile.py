import numpy as np 
from scipy.stats import vonmises
import matplotlib.pyplot as plt
from sklearn import decomposition
import sys
from ccf_interpolate import ccf
from numpy.random import default_rng

class fake_profile(object):
	def __init__(self):
	
		#import mjds and transition epochs
		self.mjds = np.loadtxt("mjd_list_no_pepoch.dat")
		transitions = np.trunc(np.loadtxt("gleps.txt"))
		
		#initilise mjd matrix
		mjd_matrix = np.loadtxt("mjd_nudot_err.dat")
		self.nudot_mjds = mjd_matrix[:,0]
		self.nudot = mjd_matrix[:,1]
		
		#xvalues to imput into von-mises
		x = np.linspace(235,275,510)
		peak_x_value = 255
		scale = 7 #determines width of von-mises
		kappa_1 = 1 #determines height
		kappa_2 = 5

		#create an array for a one component von-mises function
		single_peak = vonmises.pdf(x, kappa_1, peak_x_value, scale)
		plt.plot(x, single_peak)
		plt.show()

		#create an array for a two component von-mises function
		one_peak = vonmises.pdf(x, kappa_2, peak_x_value, scale) 
		another_peak = vonmises.pdf(x, kappa_2, 245, 10)
		double_peak = one_peak + another_peak
		plt.plot(x, one_peak)
		plt.plot(x, another_peak)
		plt.plot(x, double_peak)
		plt.show()

		#create matrix of profile values 
		profile_matrix = np.empty((len(self.nudot),0))
		j=0
		k=0

		for i in range(len(self.mjds)):
			if self.mjds[i] == transitions[k]:

				if j == 0:
					if i == 0:
						profile_matrix = double_peak
					else: 
						profile_matrix = np.vstack((profile_matrix, double_peak))
						#print("transition to double")
					k+=1
					j = 1
					continue
				elif j == 1:
					profile_matrix = np.vstack((profile_matrix, single_peak))
					#print("transition to single")
					j = 0
					k+=1
					continue
				#else:
					#print("There has been an error")
			else:
				if j == 1:
					profile_matrix = np.vstack((profile_matrix, double_peak))
					#print("double")
				if j == 0:
					if i == 0:
						profile_matrix = single_peak
					else:
						profile_matrix = np.vstack((profile_matrix, single_peak))
					#print("single")
		self.profile_matrix = profile_matrix
		
	def pca(self):
		#do pca
		N=1
		pca = decomposition.PCA(n_components=N)
		pca.fit(self.profile_matrix)
		projected = pca.transform(self.profile_matrix)

		#plot outcome
		x = np.linspace(235,275,len(pca.mean_))
		plt.plot(x,pca.mean_)
		plt.show()
		x = np.linspace(235,275,len(pca.components_[0,:]))
		plt.plot(x,pca.components_[0,:])
		plt.show()
		x=np.linspace(235,275,len(projected))
		plt.plot(x,projected)
		plt.show()
		return pca.mean_,pca.components_,projected,x
		
	def random_removal(self, coeff, n):
		#get indexes to randomly sample
		rng = default_rng()
		max_index = int(len(coeff))
		size = int(len(coeff)*(1 - 1/n))
		indexes = rng.choice(max_index, size=size, replace=False)
		
		#get coeff array, mjd array and nudot array with new indexes
		nudot_rand = np.delete(self.nudot, indexes)
		coeff_rand = np.delete(coeff, indexes)
		mjd_rand = np.delete(self.nudot_mjds, indexes)
		
		#np.savetxt("indexes_to_remove.dat", indexes)
		
		return nudot_rand, coeff_rand, mjd_rand
		
	def check_index_spacing(self, mjds, name):
		#mjds = mjds.sort()
		mjd_steps = np.array(())
		for i in range(len(mjds)-1):
			mjd_steps = np.append(mjd_steps, mjds[i+1]-mjds[i])
		average = np.mean(mjd_steps)
		print(average)
		plt.figure()
		plt.hist(mjd_steps, bins =np.arange(153))
		plt.axvline(x=average, ymin =0,ymax = 1, color = "k", linewidth = 0.5)
		plt.xlabel("time between observations (days)")
		plt.ylabel("freq.")
		plt.savefig(name)
		plt.show()
		
	def make_lags(self, lag_number):
		# lags = delta_t * k - k is an int, delta_t is constant#
		
		x = int(lag_number/2)
		k = range(-x,x+1,8)
		delta_t = 1
		lags = np.array(())
		for n in k:
			lags = np.append(lags, n*delta_t)
		
		return lags
		
	def main(self):
	
		lags = self.make_lags(2000)
		#print(lags)
		mean, components, coeff, x = self.pca()
		#randomly remove data points

		#nudot_rand, coeff_rand, mjd_rand = self.random_removal(coeff[:,0],2)
		#np.savetxt("mjds_randomly_removed.dat", mjd_rand)
		#self.check_index_spacing(mjd_rand, "index_spacing_half_removed.png")
		#do ccf
		#lags = np.arange(-np.around(np.around(np.max(mjd_rand)-np.min(self.nudot_mjds))),np.around(np.around(np.max(mjd_rand)-np.min(mjd_rand))),30)/4.5

		#cc_coeff = ccf(nudot_rand, coeff_rand, mjd_rand, mjd_rand, method = "gaussian", lags=lags, min_weighting=10, max_gap =40).results()
		cc_coeff = ccf(self.nudot, coeff[:,0], self.nudot_mjds, self.nudot_mjds, method = "gaussian", lags=lags, min_weighting=10, max_gap =40).results()
		cc_coeff_array = cc_coeff[:,1]
		cc_coeff_err = cc_coeff[:,2]
		plot_lags = cc_coeff[:,0]
		#np.savetxt("gaussian_new.dat", cc_coeff)
		plt.errorbar(plot_lags, cc_coeff_array, yerr=cc_coeff_err)
		plt.show()

do_a_thing = fake_profile()
do_a_thing.main()
