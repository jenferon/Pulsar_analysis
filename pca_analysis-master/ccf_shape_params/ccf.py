import numpy as np
import logging
from numba import jit

class ccf(object):
	def __init__(self, x, y, t_x, t_y, method = "gaussian", lags=0, min_weighting = 0,  max_gap=0, bin_width=0):
		#NOTE: ONLY BIN_WIDTH OR MAX_GAP NEED TO BE INPUTTED - max_gap used to auto calc bin_width
		#x & y np.ndarray signals to be analysed (do not need to be equal in length)
		#t_x & t_y np.ndarray times (arb unit) corresponding to values for x and y
		#method str method of weighting used (gaussian)
		#min_weighting int/float the minimum weighting before a ccf value is dicarded
		#bin_width int/float determines the temporal size of the weighting (same units as t_x and t_y)
		#max_gap max gap to be included in calculation of mean spacing - used to calculate bin_width auto
		#lags numpy np.ndarray/int/float array of desired lags or a single value of lag
		logging.basicConfig(level=logging.INFO)
		if not isinstance(x,np.ndarray):
			raise TypeError("variable x is not a numpy array")
		else:
			self.x = self.normalise(x)
			
		if not isinstance(y,np.ndarray):
			raise TypeError("variable y is not a numpy array")
		else:
			self.y = self.normalise(y)
			
		if not isinstance(t_x,np.ndarray):
			raise TypeError("variable t_x is not a numpy array")
		else:
			self.t_x = t_x
			
		if len(self.t_x) != len(self.x):
			raise TypeError("variables x & t_x are not equal lengths") 
			
		if not isinstance(t_y,np.ndarray):
			raise TypeError("variable t_y is not a numpy array")
		else:
			self.t_y = t_y
		
		if len(self.t_y) != len(self.y):
			raise TypeError("variables y & t_y are not equal lengths")
			
		if isinstance(min_weighting, int) and isinstance(min_weighting, float):
			raise TypeError("variable min_weighting is not a int/float")
		else:
			self.min_weighting = min_weighting
			
		if bin_width == 0 and max_gap ==0:
			raise SyntaxError("either bin_width or max_gap must have an entered value")
			
		if not isinstance(bin_width, float) and not isinstance(bin_width, int):
			raise TypeError("variable bin_width is not a float or int")
		else:
			self.bin_width = bin_width
			
		if not isinstance(max_gap,float) and not isinstance(max_gap, int):
			raise TypeError("variable bin_width is not a float or int")
		else:
			self.max_gap = max_gap 
			
		if not isinstance(lags,np.ndarray) and not isinstance(lags, float) and not isinstance(lags, int):
			raise TypeError("variable lags is not float, int or numpy array")
		else: 
			self.lags = lags
			
		if method != "gaussian":
			raise SyntaxError("method not implemented")
		else:
			self.method = method
			
		if self.max_gap !=0 and self.bin_width !=0:
			logging.info("WARNING: you have inputted values for max_gap and bin_width - bin_width will be overwirtten by calculation")
		
		if max_gap != 0:
			dt_mu_x = self.mean_gap(self.t_x, self.max_gap)
			dt_mu_y = self.mean_gap(self.t_y, self.max_gap)
			if dt_mu_x >= dt_mu_y:
				if self.method == "gaussian":
					self.bin_width = 0.25*dt_mu_x
			else:
				if self.method =="gaussian":
					self.bin_width = 0.25*dt_mu_y
					
	def normalise(self,x):
		#normalises np.ndarray, x, by mean
		return (x - np.mean(x))/np.std(x)
		
	def mean_gap(self, t, max_gap):
		#determines mean gap between t, exludes any values greater than max gap
		#t np.ndarray times to be analysed
		#max_gap int/float max time step to be included
		dt = np.array(())
		for i in range(len(t)-1):
			dt = np.append(dt, np.abs(t[i] - t[i+1]))
		remaining_args = np.where(dt< max_gap)
		filtered_dt = dt[remaining_args]
		return np.mean(filtered_dt)
		
	def calc_ccf(self, x, y, t_x, t_y, bin_width, lags):
		#determines method for weighthed CCF and performs correct calculation
		#x & y np.ndarray signals to be analysed (do not need to be equal in length)
		#t_x & t_y np.ndarray times (arb unit) corresponding to values for x and y
		#bin_width int/float determines the temporal size of the weighting (same units as t_x and t_y)
		#lags numpy np.ndarray/int/float array of desired lags or a single value of lag
		#returns a 2D array of lag, ccf, cum_sum_weighting (1D for 1 value of lag)
		 
		if self.method == "gaussian":
			if isinstance(lags,int) or isinstance(lags,float):#computes ccf for single lag
				ccf_results = self.calc_ccf_gaussian(x,y,t_x,t_y,bin_width,lags)
			else:
				for k in range(len(lags)):#computes ccf for array of lags
					if k ==0:#initilises variable
						ccf_results = self.calc_ccf_gaussian(x, y, t_x, t_y, bin_width, lags[k])
					else:#vstacks to initilised varaible
						ccf_results = np.vstack((ccf_results, self.calc_ccf_gaussian(x, y, t_x, t_y, bin_width, lags[k])))
		return ccf_results
	@staticmethod
	@jit(nopython=True)
	def calc_ccf_gaussian(x, y, t_x, t_y, bin_width, lag): 
		#calculates the ccf using gaussian weighting for one lag 
		#x & y np.ndarray signals to be analysed (do not need to be equal in length)
		#t_x & t_y np.ndarray times (arb unit) corresponding to values for x and y
		#bin_width int/float determines the temporal size of the weighting (same units as t_x and t_y)
		#lag int/float of desired lag
		#returns a 1D np.ndarray of lag, ccf, cum_sum_weighting
		weighting_sum = 0 #sum of weighting
		c_ij = 0 #sum of x[i]*y[j]*weighting[ij]
		for i in range(len(x)):
			for j in range(len(y)):
				t_diff = t_x[i]-t_y[j]-lag #time difference factoring in lag
				weighting = (1/np.sqrt(2*np.pi*bin_width))*np.exp(-t_diff**2/(2*bin_width**2))#gaussian weighting

				c_ij = c_ij + x[i]*y[j]*weighting
				weighting_sum = weighting_sum + weighting
		if weighting_sum == 0:#no weighting -devide by zero error

			result = np.array((lag, 0, 0, weighting_sum/2))
		else:	
			ccf = c_ij/weighting_sum #ccf = sum of x[i]*y[j]*weighting[ij]/sum of weighting
			result = np.array((lag, ccf, 0, weighting_sum/2))

		return result
			
	def error(self, x, y_in, t_x, t_y, n):
		#calculates error for ccf calculations, by calculating the ccf at 0 lags for randomly
		#scrambled arrays. This function itterates over this n times and averages to obtain an
		#error. 
		#x & y np.ndarray signals to be analysed (do not need to be equal in length)
		#t_x & t_y np.ndarray times (arb unit) corresponding to values for x and y
		#n is a float that defines the number of itterations used to calculate the error
		#returns a float equal to the error that can be used for every point in the ccf calculation
		ccf_array = np.array(())
		for i in range(n):
			y = y_in
			rng = np.random.default_rng()
			rng.shuffle(y)
			result = self.calc_ccf(x, y, t_x, t_y, self.bin_width, 0)
			ccf_array =  np.append(ccf_array, result[1])
		error = np.std(ccf_array)
		return error

	def filter_result(self, ccf_array, min_weighting):
		#filters the ccf results by weighting against min weighting
		#ccf_array 2D array of [lags, ccf, cum_sum_weighting]
		#min_weighting int/float the minimum weighting required before the result is discarded
		#print(ccf_array)
		keep_args = np.where(ccf_array[:,3]>=min_weighting)

		return ccf_array[keep_args[0],:]
		
	def results(self):
		#returns result of ccf on inputs 
		unfiltered_ccf = self.calc_ccf(self.x, self.y, self.t_x, self.t_y, self.bin_width, self.lags)
		filtered_ccf = self.filter_result(unfiltered_ccf, self.min_weighting)
		error_iterations = 10000
		error = self.error(self.x, self.y, self.t_x, self.t_y, error_iterations)
		filtered_ccf[:,2]=error
		print(error)

		return filtered_ccf
		
					

