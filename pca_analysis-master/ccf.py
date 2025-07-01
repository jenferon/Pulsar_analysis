#performs ccf analysis on time series with unequal, irregular sampling using weighting 
#currently gaussian, sinc, rectangle kernal and regular ccf implemented 
#requires numba installed
#NOTE: ONLY BIN_WIDTH OR MAX_GAP NEED TO BE INPUTTED - max_gap used to auto calc bin_width
#NOTE: For standard method can only calculate 1 lag if the lag is equal to zero
import numpy as np
import logging
from numba import jit



class ccf(object):
	def __init__(self, x, y, t_x, t_y, method = "gaussian", lags=0, min_weighting = 0,  max_gap=0, bin_width=0, confidence = 6e-7):
		#NOTE: ONLY BIN_WIDTH OR MAX_GAP NEED TO BE INPUTTED - max_gap used to auto calc bin_width
		#x & y np.ndarray signals to be analysed (do not need to be equal in length)
		#t_x & t_y np.ndarray times (arb unit) corresponding to values for x and y
		#method str method of weighting used (gaussian, sinc, retangle or standard (for evenly sampled data - this method requires evenly spaced lags, others do not)
		#min_weighting int/float the minimum weighting before a ccf value is dicarded
		#bin_width int/float determines the temporal width of the weighting (same units as t_x and t_y)
		#max_gap max gap to be included in calculation of mean spacing - used to calculate bin_width auto
		#lags numpy np.ndarray/int/float array of desired lags or a single value of lag
		#confidence - when checking if two values are equal, allows a range of values to account for rounding errors- default is 5 sigma (6e-7)
		logging.basicConfig(level=logging.INFO)
		logging.debug("entering initiliser")
		if not isinstance(x,np.ndarray):
			raise TypeError("variable x is not a numpy array")
		else:
			if len(x)==0:
				raise TypeError("An empty array has been entered for x")
			self.x = self.normalise(x)#x normalised when initialised as a class variable
			
		if not isinstance(y,np.ndarray):
			raise TypeError("variable y is not a numpy array")
		else:
			if len(y)==0:
				raise TypeError("An empty array has been entered for y")
			self.y = self.normalise(y)#y normalised when initialised as a class variable
			
		if not isinstance(t_x,np.ndarray):
			raise TypeError("variable t_x is not a numpy array")
		else:
			if len(t_x)==0:
				raise TypeError("An empty array has been entered for t_x")
			self.t_x = t_x
			
		if len(self.t_x) != len(self.x):
			raise TypeError("variables x & t_x are not equal lengths") 
			
		if not isinstance(t_y,np.ndarray):
			raise TypeError("variable t_y is not a numpy array")
		else:
			if len(t_y)==0:
				raise TypeError("An empty array has been entered for t_y") 
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
			if isinstance(lags,np.ndarray):
				if len(lags)==0:
					raise TypeError("An empty array has been entered for lags") 
			self.lags = lags
		if not isinstance(confidence, float):
			raise TypeError("variable confidence is not float")
		else:
			self.confidence = confidence
		if method != "gaussian" and method != "standard" and method != "sinc" and method != "rectangle" and method !="triangle":
			raise SyntaxError("method not implemented")
		else:
			self.method = method
			
		if self.max_gap !=0 and self.bin_width !=0:
			logging.info("WARNING: you have inputted values for max_gap and bin_width - bin_width will be overwirtten by calculation")
		
		if max_gap != 0:#calculates h (maximum mean time spacing between x and y (excludes spacings above max gap from calc)
			h = np.max(np.array((self.mean_gap(self.t_x, self.max_gap),self.mean_gap(self.t_y, self.max_gap))))
			if self.method =="gaussian":
				self.bin_width = 0.25*h
			elif self.method =="standard":
				self.bin_width = 0
			elif self.method =="sinc":
				self.bin_width = h
			elif self.method=="rectangle":
				self.bin_width = h*0.5
			elif self.method=="triangle":
				self.bin_width = h*0.5
		logging.debug("exiting initiliser")

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
		elif self.method == "rectangle":
			if isinstance(lags,int) or isinstance(lags,float):#computes ccf for single lag
				ccf_results = self.calc_ccf_rectangle(x,y,t_x,t_y,bin_width,lags)
			else:
				for k in range(len(lags)):#computes ccf for array of lags
					if k ==0:#initilises variable
						ccf_results = self.calc_ccf_rectangle(x, y, t_x, t_y, bin_width, lags[k])
					else:#vstacks to initilised varaible
						ccf_results = np.vstack((ccf_results, self.calc_ccf_rectangle(x, y, t_x, t_y, bin_width, lags[k])))
		elif self.method == "sinc":
			if isinstance(lags,int) or isinstance(lags,float):#computes ccf for single lag
				ccf_results = self.calc_ccf_sinc(x,y,t_x,t_y,bin_width,lags)
			else:
				for k in range(len(lags)):#computes ccf for array of lags
					if k ==0:#initilises variable
						ccf_results = self.calc_ccf_sinc(x, y, t_x, t_y, bin_width, lags[k])
					else:#vstacks to initilised varaible
						ccf_results = np.vstack((ccf_results, self.calc_ccf_sinc(x, y, t_x, t_y, bin_width, lags[k])))
		elif self.method == "standard":
			if isinstance(lags,int) or isinstance(lags,float):#computes standard ccf for single lag
				if lags !=0:
					raise TypeError("when using method = standard for 1 lag, only lag=0 can be calculated")
				ccf_results = self.calc_ccf_standard(x,y,0,0)
			else:
				lag_numbers = self.calc_lag_numbers(lags,t_x,t_y)
				for k in range(len(lags)):#computes ccf for array of lags
					if k ==0:#initilises variable
						ccf_results = self.calc_ccf_standard(x, y,lags[k], lag_numbers[k])
					else:#vstacks to initilised varaible
						ccf_results = np.vstack((ccf_results, self.calc_ccf_standard(x, y,lags[k], lag_numbers[k])))
		elif self.method == "triangle":
			if isinstance(lags,int) or isinstance(lags,float):#computes ccf for single lag
				ccf_results = self.calc_ccf_triangle(x,y,t_x,t_y,bin_width,lags)
			else:
				for k in range(len(lags)):#computes ccf for array of lags
					if k ==0:#initilises variable
						ccf_results = self.calc_ccf_triangle(x, y, t_x, t_y, bin_width, lags[k])
					else:#vstacks to initilised varaible
						ccf_results = np.vstack((ccf_results, self.calc_ccf_triangle(x, y, t_x, t_y, bin_width, lags[k])))
		return ccf_results
	def check_constant_spacing(self, array):
		#checks if an array has equal spacing between each consecutive values
		#NOTE - checks spacing to 5 sigma (99.99994%) to account for rounding
		# array - ndarray to be tested
		#returns bool
		if len(array) < 2:
		    raise TypeError("cannot compute ccf when lags is an array with length 1")
		
		diff_control = np.abs(array[0] - array[1])
		for i in range(len(array)-1):
		    if diff_control < np.abs(array[i] - array[i+1])*(1-self.confidence) or diff_control > np.abs(array[i] - array[i+1])*(1+self.confidence):
		        return False
		return True

	def calc_spacing(self, t_x, t_y):
		#calculates spacing between points, calls check_constant_spacing and checks if t_x and t_y are the same - used for standard method
		#t_x & t_y ndarray of ints/ floats
		if not self.check_constant_spacing(t_x) and not self.check_constant_spacing(t_y):
		        raise TypeError("when using standard method, times must follow k*lags_spacing (k is a set of integers iteratively increasing by 1")
		if t_x.all() != t_y.all():
			raise TypeError("when using standard method, times for x and y must be equal")
		t_spacings = np.array(())
		for i in range(len(t_x)-1):#for loop to help with rounding errors
			t_spacings = np.append(t_spacings, np.abs(t_x[i]-t_x[i+1]))
		return np.mean(t_spacings)

	def calc_lag_numbers(self, lags, t_x, t_y):
		#checks lags follow k*dt (k is a set of integers & dt is ) and calculates the lag number for each lag in the array
		#lags - ndarray
		#t_x & t_y ndarray of ints/ floats
		#returns - ndarray of lag numbers
		lag_numbers = np.array((),dtype=int)
		lag_spacing = self.calc_spacing(t_x, t_y)
		if len(lags) == len(t_x):
			raise TypeError("when using standard method, number of x points must be larger than the number of lags") 
		for i in range(len(lags)):
			lag_no = lags[i]/lag_spacing
			if np.abs(lag_no - round(lag_no))<self.confidence: 
				lag_numbers =  np.append(lag_numbers,int(round(lag_no)))
			else:	
				raise TypeError("when using standard method, lags must follow k*lags_spacing (k is a set of integers iteratively increasing by 1")
		return lag_numbers
		    
			
	@staticmethod
	@jit(nopython=True)
	def calc_ccf_gaussian(x, y, t_x, t_y, bin_width, lag): 
		#calculates the ccf using gaussian weighting for one lag 
		#x & y np.ndarray signals to be analysed (do not need to be equal in length)
		#t_x & t_y np.ndarray times (arb unit) corresponding to values for x and y
		#bin_width int/float determines the temporal size of the weighting (same units as t_x and t_y)
		#lag int/float of desired lag
		#returns a 1D np.ndarray of lag, ccf, 0 (placeholder for error), cum_sum_weighting
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
			ccf = c_ij/weighting_sum #ccf = sum of x[i]*y[j]*weighting[ij]/sum of weighting (pre-normalised x&y = (x&y - mean{x&y})/std{x&y})
			result = np.array((lag, ccf, 0, weighting_sum/2))

		return result
		
	@staticmethod
	@jit(nopython=True)
	def calc_ccf_rectangle(x, y, t_x, t_y, bin_width, lag): 
		#calculates the ccf using rectangle weighting for one lag 
		#x & y np.ndarray signals to be analysed (do not need to be equal in length)
		#t_x & t_y np.ndarray times (arb unit) corresponding to values for x and y
		#bin_width int/float determines the temporal size of the weighting (same units as t_x and t_y)
		#lag int/float of desired lag
		#returns a 1D np.ndarray of lag, ccf, 0 (placeholder for error), cum_sum_weighting
		weighting_sum = 0 #sum of weighting
		c_ij = 0 #sum of x[i]*y[j]*weighting[ij]
		for i in range(len(x)):
			for j in range(len(y)):
				t_diff = t_x[i]-t_y[j]-lag #time difference factoring in lag
				if np.abs(t_diff) <= bin_width:#rectangle weighting
					weighting = 1
				else:
					weighting = 0
				c_ij = c_ij + x[i]*y[j]*weighting
				weighting_sum = weighting_sum + weighting
		if weighting_sum == 0:#no weighting -devide by zero error

			result = np.array((lag, 0, 0, weighting_sum/2))
		else:	
			ccf = c_ij/weighting_sum #ccf = sum of x[i]*y[j]*weighting[ij]/sum of weighting (pre-normalised x&y = (x&y - mean{x&y})/std{x&y})
			result = np.array((lag, ccf, 0, weighting_sum/2))		
		return result
	@staticmethod
	@jit(nopython=True)
	def calc_ccf_triangle(x, y, t_x, t_y, bin_width, lag): 
		#calculates the ccf using triangle weighting for one lag 
		#x & y np.ndarray signals to be analysed (do not need to be equal in length)
		#t_x & t_y np.ndarray times (arb unit) corresponding to values for x and y
		#bin_width int/float determines the temporal size of the weighting (same units as t_x and t_y)
		#lag int/float of desired lag
		#returns a 1D np.ndarray of lag, ccf, 0 (placeholder for error), cum_sum_weighting
		weighting_sum = 0 #sum of weighting
		c_ij = 0 #sum of x[i]*y[j]*weighting[ij]
		for i in range(len(x)):
			for j in range(len(y)):
				t_diff = t_x[i]-t_y[j]-lag #time difference factoring in lag
				if np.abs(t_diff) <= bin_width:#rectangle weighting
					weighting = 1-np.abs(t_diff)/bin_width
				else:
					weighting = 0
				c_ij = c_ij + x[i]*y[j]*weighting
				weighting_sum = weighting_sum + weighting
		if weighting_sum == 0:#no weighting -devide by zero error

			result = np.array((lag, 0, 0, weighting_sum/2))
		else:	
			ccf = c_ij/weighting_sum #ccf = sum of x[i]*y[j]*weighting[ij]/sum of weighting (pre-normalised x&y = (x&y - mean{x&y})/std{x&y})
			result = np.array((lag, ccf, 0, weighting_sum/2))		
		return result
	@staticmethod
	def calc_ccf_standard(x, y,lag, lag_no): 
		#calculates the ccf using the standard method (for regularly sampled data)
		#x & y np.ndarray signals to be analysed (do not need to be equal in length)
		#lag int/float of desired lag
		#returns a 1D np.ndarray of lag, ccf, 0 (placeholder for error), weighting
		c_ij = 0 #sum of x[i]*y[j]*weighting[ij]
		N=len(x)-np.abs(lag_no)
		for i in range(N):
			c_ij = c_ij + x[i]*y[i+lag_no]
		if len(x)-lag_no == 0:#no weighting -devide by zero error

			result = np.array((lag, 0, 0, 0))
		else:	
			ccf = c_ij/N #ccf = sum of x[i]*y[j]/(N-k) (pre-normalised x&y = (x&y - mean{x&y})/std{x&y})
			result = np.array((lag, ccf, float(0), float(N)))
		return result
	
	@staticmethod
	@jit(nopython=True)
	def calc_ccf_sinc(x, y, t_x, t_y, bin_width, lag): 
		#calculates the ccf using sinc weighting for one lag 
		#x & y np.ndarray signals to be analysed (do not need to be equal in length)
		#t_x & t_y np.ndarray times (arb unit) corresponding to values for x and y
		#bin_width int/float determines the temporal size of the weighting (same units as t_x and t_y)
		#lag int/float of desired lag
		#returns a 1D np.ndarray of lag, ccf, 0 (placeholder for error), cum_sum_weighting
		weighting_sum = 0 #sum of weighting
		c_ij = 0 #sum of x[i]*y[j]*weighting[ij]
		K = np.sqrt(len(x)*len(y))*np.pi*bin_width
		for i in range(len(x)):
			for j in range(len(y)):
				t_diff = t_x[i]-t_y[j]-lag #time difference factoring in lag
				
				if t_diff == 0:
					c_ij = c_ij + np.sqrt(len(x)*len(y))*x[i]*y[j]
					weighting_sum = weighting_sum + np.sqrt(len(x)*len(y))
				else:
					a = K*t_diff
					weighting = np.abs((1/a)*np.sin(np.pi*bin_width*t_diff))#sinc weighting
					c_ij = c_ij + x[i]*y[j]*weighting
					weighting_sum = weighting_sum + weighting
		if weighting_sum == 0:#no weighting -devide by zero error

			result = np.array((lag, 0, 0, weighting_sum/2))
		else:	
			ccf = c_ij/weighting_sum #ccf = sum of x[i]*y[j]*weighting[ij]/sum of weighting (pre-normalised x&y = (x&y - mean{x&y})/std{x&y})
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
		if np.shape(ccf_array)[1]<1:
			if ccf_array[3] >=min_weighting:
				return ccf_array[3]
			else:
				logging.info("Warning: All weightings below inputted threshold")
		else:
			keep_args = np.where(ccf_array[:,3]>=min_weighting)
			if len(keep_args[0])==0:
				logging.info("Warning: All weightings below inputted threshold")
			return ccf_array[keep_args[0],:]
		
	def results(self):
		#returns result of ccf on inputs 
		unfiltered_ccf = self.calc_ccf(self.x, self.y, self.t_x, self.t_y, self.bin_width, self.lags)
		filtered_ccf = self.filter_result(unfiltered_ccf, self.min_weighting)
		error_iterations = 10000
		error = self.error(self.x, self.y, self.t_x, self.t_y, error_iterations)
		filtered_ccf[:,2] = error
		return filtered_ccf
