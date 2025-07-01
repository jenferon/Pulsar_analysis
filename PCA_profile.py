#pulse profile analysis
import pulsarpvc as pp
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
import logging
from ccf_interpolate import ccf
import matplotlib as mpl


class PCA(object):
	def __init__(self, input_matrix, bad_mjds, nudot_matrix_str, N, start=0, end=0, nudot_file_names = "", profile_file_names="",start_cut_epoch=0, debug = False):
		#initilises
		plt.set_loglevel("info")
		
		if not debug:
		    logging.basicConfig(level=logging.INFO)
		else:
		    logging.basicConfig(level=logging.DEBUG)
		  
		logging.debug("Initilising Object")
		
		nudot_matrix = np.loadtxt(nudot_matrix_str)
		
		self.input_matrix = input_matrix #filename of the input matrix (str)
		self.bad_mjds = bad_mjds #filename of bad MJDs (str)
		self.nudot = nudot_matrix[:,1] #spin down variations
		self.nudot_mjds = nudot_matrix[:,0] #mjds corrosponding to spin down
		self.nudot_err = nudot_matrix[:,2] #error on values of spin down 
		self.N = N #Number of components for PCA (int) or 'mle' allow code to estimate optimimum N
		self.start = start # start and end defines on pulse region
		self.end = end
		self.start_cut_epoch = start_cut_epoch#defines a cut off for the coeff plot
		self.nudot_file_names = nudot_file_names
		self.profile_file_names = profile_file_names
        
	def prepare_data(self, input_matrix, bad_mjds, start, end):
		#prepares data to be analysed by the PCA
		#create preprocessor and pass in our data array
		logging.debug("entering prepare_data")
		preproc = pp.Preprocess(input_matrix)

		#unflitered raw data
		raw_data = preproc.rawdata #1026 by no of obvs
		
		
		name_indexes = np.array(())
		
		original_mjds = raw_data[np.shape(raw_data)[0]-1,:]

		#find index of bad mjds
		
		
		#filter bad mjds (put this in after first iteration
		filtered = preproc.filter(bad_mjds)

		#split up data
		data, mjds, tobd, bins = preproc.stripinput(filtered)

		for i in range(len(mjds)):
			name_indexes = np.append(name_indexes, np.argwhere(original_mjds==mjds[i])[0])
		
		#debase profiles
		debased ,removed_profiles, rms_per_epoch, outliers, inliers = preproc.debase(data)
		print(np.median(rms_per_epoch))
		print(len(rms_per_epoch))
		#defining outlier MJDs 
		outlier_mjds = np.delete(mjds, inliers)
		
		removed_outliers = np.array(())
		for i in range(len(outlier_mjds)):
			index_value = np.argwhere(original_mjds == outlier_mjds[i])[0]
			arg_to_be_deleted = np.argwhere(name_indexes == index_value)
			name_indexes = np.delete(name_indexes, arg_to_be_deleted)

		
		#adjust mjd list to remove outliers 
		mjds =np.delete(mjds, outliers)

		#find index of brightest pulse
		index_of_brightest =preproc.find_brightest(debased, rms_per_epoch)

		#align data to form template

		aligned, brightest, template = preproc.align(debased, index_of_brightest)
		
		if start != 0 and end !=0:
			#select on pulse region
			logging.debug("cutting data")
			to_be_rotated = aligned[start:end,:]
		else:
			logging.debug("no on pulse region defined: not cutting data")
			to_be_rotated = aligned
			self.start =0
			self.end = np.shape(to_be_rotated)[0]
		
		#normalise data (comment out dependent on which normalise method is prefered
		to_be_rotated = self.normalise_mean(to_be_rotated) #normalise with mean
		#to_be_rotated = self.normalise_max(to_be_rotated) #normalise with max power
		
		#rotate matrix to be used in PCA
		final_data = list(zip(*to_be_rotated[::1]))
		self.profiles = to_be_rotated
		self.data = final_data
		self.mjds = mjds
		np.savetxt("profile_mjds.dat", mjds)
		self.name_indexes = name_indexes.astype(int)
		np.savetxt("name_indexes.dat", name_indexes)
		#debug info
		
		logging.debug("removed_outliers shape = "+str(np.shape(outliers)))
		logging.debug("name_indexes shape = "+str(np.shape(name_indexes)))
		logging.debug("input matrix shape = "+str(np.shape(raw_data)))
		logging.debug("final matrix shape = "+str(np.shape(final_data)))
		logging.debug("# removed profiles = {}".format(len(outliers)))
		logging.debug("# kept profiles = {}".format(len(inliers)))
		logging.debug("exiting prepare_data")
		

	def normalise_mean(self, data):
		#normalise the data using the mean
		logging.debug("Entered Normaliser method")
		
		nprofiles = data.shape[1]
		
		for i in range(nprofiles):
			data[:,i] = data[:,i] / np.mean(data[:,i])
			
		return data
        

	def profile_nudot_matching(self,nudot_names_file, profile_names_file):
		#match nudot values to profiles
		nudot_names = np.loadtxt(nudot_names_file,dtype=str)
		profile_names = np.loadtxt(profile_names_file,dtype=str)
		filtered_profile_names = profile_names[self.name_indexes.astype(int)]
		
		nudot_indexes = np.array((), dtype=int)
		for i in range(len(filtered_profile_names)):
			args = np.argwhere(nudot_names==filtered_profile_names[i])
			if len(args) !=0:
				nudot_indexes = np.append(nudot_indexes, args)
			elif len(args)>1:
				logging.info("WARNING-- duplicate profile names in "+nudot_names_file)
		
		filtered_nudot_names = nudot_names[nudot_indexes]
		profile_indexes = np.array((), dtype=int)
		for i in range(len(filtered_nudot_names)):
			args = np.argwhere(filtered_profile_names==filtered_nudot_names[i])
			if len(args) !=0:
				profile_indexes = np.append(profile_indexes, args)
			elif len(args)>1:
				logging.info("WARNING-- duplicate profile names in "+profile_names_file)
		return nudot_indexes, profile_indexes
			
	


	def normalise_max(self, data):
		#normalise data using maximum power
		logging.debug("Entered Normaliser method")
		
		nprofiles = data.shape[1]
		
		for i in range(nprofiles):
			data[:,i] = data[:,i] / np.max(data[:,i])
			
		return data
        
	def plot_N_against_CEV(self):
		#creates graph for choosing number of components
		logging.debug("plotting CEV graph")
		pca = decomposition.PCA().fit(self.data)
		plt.plot(np.cumsum(pca.explained_variance_ratio_))
		plt.xlabel('number of components')
		plt.ylabel('cumulative explained variance')
		plt.savefig("PCA_outputs/number_of_components_graph.png")
		plt.show()

	def stacked_plotter(self, x, y, labels, x_label, y_label, name):
		for i in range(np.shape(y)[0]):
			plt.plot(x,y[i], label=labels[i])
		plt.legend()
		plt.xlabel(x_label)
		plt.ylabel(y_label)	
		plt.savefig("PCA_outputs/"+name)
		plt.show()
		
	def sub_plot(self, x, y, x_label, y_label, name):
		#creates subplots
		try:
			np.shape(y)[1]
			fig = plt.figure()

			tot = np.shape(y)[0]
			cols = 1

			# Compute Rows required

			rows = tot // cols 
			rows += tot % cols

			# Create a Position index

			position = range(1,tot + 1)
			for i in range(np.shape(y)[0]):
	  			ax = fig.add_subplot(rows, cols, position[i])
	  			ax.plot(x,y[i])
	  			ax.set_ylabel(y_label[i])
	  			ax.get_yaxis().set_label_coords(-0.1,0.5)
	  			if i != np.shape(y)[0]-1:
	  				ax.xaxis.set_ticklabels([])
			ax.set_xlabel(x_label)
			plt.subplots_adjust(hspace=.0)
			plt.savefig("PCA_outputs/"+name)
		except IndexError:
				fig, ax = plt.subplots()
				ax.plot(x,y)
				ax.set_ylabel(y_label[0])
		ax.set_xlabel(x_label)
		plt.subplots_adjust(hspace=.0)
		plt.savefig("PCA_outputs/"+name)
		plt.show()	



	def sub_plot_err(self, x, y, err, x_label, y_label, name):
		#subplot for with errors
		
		try:
			np.shape(y)[1]
			fig = plt.figure()

			tot = np.shape(y)[0]
			cols = 1

			# Compute Rows required

			rows = tot // cols 
			rows += tot % cols

			# Create a Position index

			position = range(1,tot + 1) 
			for i in range(np.shape(y)[0]):
				ax = fig.add_subplot(rows, cols, position[i])
				ax.errorbar(x,y[i],err[i],color = "k", ecolor= "red", elinewidth= 0.5)
				ax.set_ylabel(y_label[i])
				ax.get_yaxis().set_label_coords(-0.1,0.5)
				if i != np.shape(y)[0]-1:
					ax.xaxis.set_ticklabels([])
		except IndexError:
				fig, ax = plt.subplots()
				ax.errorbar(x,y,err,color = "k", ecolor= "red", elinewidth= 0.5)
				ax.set_ylabel(y_label[0])
		ax.set_xlabel(x_label)
		plt.subplots_adjust(hspace=.0)
		plt.savefig("PCA_outputs/"+name)
		plt.show()	
	def plot_coeff(self, mjds, nudot, nudot_err, profile_mjds, coeff, start_cut_epoch, name, pos_indicator = False, index = 0):
		#plot coeffcients of pca on subplots with nudot for comparison
		
		#make cut of graph
		first_mjd_arg_coeff = np.argmin(np.abs(profile_mjds-start_cut_epoch))
		first_mjd_arg_nudot = np.argmin(np.abs(mjds-start_cut_epoch))
		cut_profile_mjds = profile_mjds[first_mjd_arg_coeff :]
		cut_coeff = coeff[first_mjd_arg_coeff :,:]
		cut_nudot = nudot[first_mjd_arg_nudot :]
		cut_mjds = mjds[first_mjd_arg_nudot :]
		cut_nudot_err = nudot_err[first_mjd_arg_nudot :]
		
		
		fig = plt.figure()

		tot = np.shape(coeff)[1]+1
		cols = 1

		# Compute Rows required

		rows = tot // cols 
		rows += tot % cols

		# Create a Position index
		ax = fig.add_subplot(rows, cols, 1)
		ax.errorbar(cut_mjds, cut_nudot, yerr = cut_nudot_err, label="spin down", color='r')
		ax.set_xlim(np.min(cut_profile_mjds),np.max(cut_profile_mjds))
		for i in range(0, len(cut_mjds)):
			ax.axvline((cut_mjds[i]), ymin=0.01, ymax=0.05, linestyle='solid', color='k', linewidth=1)
		ax.set_ylabel(r"$\dot{\nu}$")
		ax.get_yaxis().set_label_coords(-0.1,0.5)
		ax.xaxis.set_ticklabels([])
		position = range(2,tot + 1)
		output = ""
		labels = np.array([r"$C_1$", r"$C_2$", r"$C_3$"])
		if pos_indicator:
			ax.axvline((profile_mjds[index]), ymin=0.01, ymax=1, linestyle='dashed', color='grey', linewidth=1, alpha =0.5)
			output += "Modified Julian Days: "+str(profile_mjds[index])
		for i in range(np.shape(coeff)[1]):
  			ax = fig.add_subplot(rows, cols, position[i])
  			ax.plot(cut_profile_mjds, cut_coeff[:,i])
  			for j in range(0, len(cut_profile_mjds)):
  				ax.axvline((cut_profile_mjds[j]), ymin=0.01, ymax=0.05, linestyle='solid', color='k', linewidth=1)
  			ax.set_ylabel(labels[i])
  			ax.get_yaxis().set_label_coords(-0.1,0.5)
  			ax.set_xlim(np.min(cut_profile_mjds),np.max(cut_profile_mjds))
  			if i != np.shape(coeff)[1]-1:
  				ax.xaxis.set_ticklabels([])
  			if pos_indicator:
  				ax.axvline((profile_mjds[index]), ymin=0.01, ymax=1, linestyle='dashed', color='grey', linewidth=1, alpha =0.5)
  				output += " coeff {} = {} ".format(i+1, coeff[index,i])
  				
  				

  				

		plt.xlabel("MJD")
		plt.subplots_adjust(hspace=.0)
		plt.savefig("PCA_outputs/"+name+".png")
		plt.show()
		if output != "":
  			logging.info(output)
		
		
	def cut_data(self):
	
		#cuts data to correct length
		
		#find the min and the max mjd to correspond to both mjd arrays
		logging.debug("entering cut_data")
		diff_min = np.abs(self.nudot_mjds - np.min(self.mjds))
		first_mjd_arg = np.argmin(diff_min)

		diff_max = np.abs(self.nudot_mjds - np.max(self.mjds))
		last_mjd_arg = np.argmin(diff_max)

		#slices the data to the min and the max mjd
		cut_mjd = self.nudot_mjds[first_mjd_arg:(last_mjd_arg+1)] 
		cut_nudot = self.nudot[first_mjd_arg:(last_mjd_arg+1)]
		cut_err = self.nudot_err[first_mjd_arg:(last_mjd_arg+1)]
		logging.debug("length: uncut = {}, cut = {}, mjd limits: bottom = {}, top = {}".format(len(self.nudot_mjds), len(cut_mjd), np.min(cut_mjd), np.max(cut_mjd)))
		logging.debug("exiting cut_data")
		
		return [cut_mjd, cut_nudot, cut_err]

	def rebuild_profile(self, mean, components, coeff, profiles, index, mjd, nudot, err):	
		rebuilt_profile = mean 
		for i in range(len(components[:,0])):
			rebuilt_profile = rebuilt_profile + coeff[index,i]*components[i,:]
		bins = np.linspace(self.start,self.end,self.end-self.start)
		plt.plot(bins, rebuilt_profile,'k', label = "reconstructed")	
		plt.plot(bins, profiles[:,index],'orange', label = "original")
		plt.plot(bins, mean,'-.', label = "mean",alpha=1)
		for i in range(len(components[:,0])):
			plt.plot(bins, coeff[index,i]*components[i,:],'-.', label = "{} dev crb.".format(i+1),alpha=1)
		plt.legend()
		plt.xlabel("Bin")
		plt.ylabel("intensity [arb.]")
		plt.title(self.mjds[index])
		plt.savefig("PCA_outputs/recovered_profiles/"+str(self.mjds[index])+"_recovered.png")
		plt.show()
		
		
	def recon_profile_residual_sum(self, mean, components, coeff, profiles):
		#shows the total residuals as a function of bin	
		resids = np.array(())
		for i in range(len(profiles[:,0])):
			rebuilt_profile = mean 
			for j in range(len(components[:,0])):
				rebuilt_profile = rebuilt_profile + coeff[i,j]*components[j,:]
			if i == 0:
				resids = profiles[:,i] - rebuilt_profile
			else:
				resids = resids + (profiles[:,i] - rebuilt_profile)
		bins = np.linspace(self.start,self.end,self.end-self.start)
		plt.plot(bins, resids,'k', label= "summed residuals")
		plt.plot(bins, mean, label="mean (for ref)")
		plt.legend()	
		plt.xlabel("Bin")
		plt.ylabel("Intensity [arb.]")
		plt.savefig("PCA_outputs/profile_reconstruction_residuals.png")
		plt.show()
		
	def find_index(self, mjds, required_mjd):
		#finds index of mjd in array closest to required mjd
		return np.argmin(np.abs(mjds-required_mjd))
	
	def plot_cc_coeff(self,nudot, coeff, profile_mjds, nudot_mjds,n):
		#organises inputs to for use in cross_correlate and plots output on subplots
		logging.debug("entering plot_cc_coeff") 
		y_label =[]
		rel_cadence = 5
		lag_lim = 1000
		min_weighting = 0
		max_gap = 40
		lags = np.arange(-lag_lim,lag_lim,rel_cadence)
		for i in range(len(coeff[0,:])):
			if i == 0:
				cc_coeff = ccf(nudot, coeff[:,i],nudot_mjds, profile_mjds, method = "gaussian", lags=lags, min_weighting=min_weighting, max_gap = max_gap).results()
				cc_coeff_array = cc_coeff[:,1]
				cc_coeff_err = cc_coeff[:,2]
				plot_lags = cc_coeff[:,0]
				weighting = cc_coeff[:,3]
				ccf_zero_lag = cc_coeff[np.where(cc_coeff[:,0]==0)[0], 1:3]
				self.weighting_checker(weighting, "_first_component")
			else:
				cc_coeff = ccf(nudot, coeff[:,i],nudot_mjds, profile_mjds, method = "gaussian", lags=lags, min_weighting=min_weighting, max_gap = max_gap).results()
				ccf_zero_lag = np.vstack((ccf_zero_lag,cc_coeff[np.where(cc_coeff[:,0]==0)[0], 1:3]))
				cc_coeff_array = np.vstack((cc_coeff_array, cc_coeff[:,1]))
				cc_coeff_err = np.vstack((cc_coeff_err,cc_coeff[:,2]))
				self.weighting_checker(weighting, "_"+str(i)+"_component")
			y_label.append(r"DCF($\dot{{\nu}}$,$C_{{{number}}}$)".format(number=i+1))
		
		np.savetxt("PCA_outputs/ccf_zero_lags_pca.txt",ccf_zero_lag)
		self.sub_plot_err(plot_lags, cc_coeff_array, cc_coeff_err, "lag (days)", y_label, "cross_correlation_coeff.png")

		logging.debug("exiting plot_cc_coeff")

	def perform_analysis(self,cut_mjd,cut_nudot,cut_err):
		logging.debug("performing analysis")
		#pca implimented
		pca = decomposition.PCA(n_components=self.N)
		pca.fit(self.data)
		projected = pca.transform(self.data)
		
		#fit information
		logging.info("Profiles found: {}  No. of bins: {}  Projected No. of bins: {} CEV: {}".format(np.shape(self.data)[0],np.shape(self.data)[1],np.shape(projected)[1],np.sum(pca.explained_variance_ratio_)))
		
		#generate stacked deviation labels
		x=np.linspace(self.start,self.end,self.end-self.start)
		labels_deviations = ["Mean",r"$1^{\mathrm{st}}$ PC", r"$2^{\mathrm{nd}}$ PC", r"$3^{\mathrm{rd}}$ PC"]
		"""for i in range(np.shape(projected)[1]):
			labels_deviations.append("Deviation [{}]".format(i+1)) 
		"""
		#plot devaitions stacted
		self.stacked_plotter(x, np.vstack((pca.mean_,pca.components_)),labels_deviations,"Bins","Intensity (arb.)", "bins_v_components.png")
		
		#generated subplot devaition labels
		"""labels_deviations = ["Mean"]
		for i in range(np.shape(projected)[1]):
			labels_deviations.append("Dev [{}]".format(i+1)) 
		"""
		#plot devaitions sub plots 
		self.sub_plot(x, np.vstack((pca.mean_,pca.components_)),"Bins",labels_deviations, "bins_v_components_sub_plots.png")

		#plot coeffs
		self.plot_coeff(cut_mjd,cut_nudot,cut_err, self.mjds, projected, self.start_cut_epoch,"coefficient_variations")
		return pca.mean_,pca.components_,projected
		
	def mjd_checker(self, mjds):
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
		plt.savefig("profile_time_step_hist.png")
		plt.show()
		
	def weighting_checker(self, weighting, name):
		average = np.mean(weighting)
		plt.figure()
		plt.hist(weighting, bins =np.arange(153))
		plt.axvline(x=average, ymin =0,ymax = 1, color = "k", linewidth = 0.5)
		plt.xlabel("weighting")
		plt.ylabel("freq.")
		plt.savefig("weighting_hist" + name)
		plt.show()
	
	def main(self):

		self.prepare_data(self.input_matrix, self.bad_mjds, self.start, self.end)
		#self.plot_N_against_CEV()
		[cut_mjd, cut_nudot, cut_err] = self.cut_data()
		print(len(cut_mjd))
		
		mean, components, coeff = self.perform_analysis(cut_mjd,cut_nudot,cut_err)		
		#self.rebuild_profile(mean, components, coeff, self.profiles, self.find_index(self.mjds, 58017), cut_mjd, cut_nudot, cut_err)
		#self.recon_profile_residual_sum(mean, components, coeff, self.profiles)
		for i in range(len(coeff[0,:])):
			np.savetxt("PCA_outputs/coeff_{}_mean_norm.dat".format(i+1), np.column_stack((self.mjds,coeff[:,i])))
		fil_nudot_indexes, fil_profile_indexes = self.profile_nudot_matching(self.nudot_file_names,self.profile_file_names)
		self.mjd_checker(self.mjds[fil_profile_indexes])
		self.plot_cc_coeff(cut_nudot[fil_nudot_indexes], coeff[fil_profile_indexes], self.mjds[fil_profile_indexes], cut_mjd[fil_nudot_indexes], 4.5)
		self.plot_coeff(cut_mjd[fil_nudot_indexes],cut_nudot[fil_nudot_indexes], cut_err[fil_nudot_indexes], self.mjds[fil_profile_indexes], coeff[fil_profile_indexes], self.start_cut_epoch, "coefficient_variations")

		

			
		


#DFB

do_a_thing = PCA("PCA_inputs/test_pulsar_matrix_1024.txt","PCA_inputs/bad_mjds.txt","PCA_inputs/nudot_matrix_final.dat",3,235,275, "PCA_inputs/tim_file_FTp_names.dat","PCA_inputs/FTp_file_names.dat") #DFB -uncut


#do_a_thing = PCA("PCA_inputs/test_pulsar_matrix_1024.txt","PCA_inputs/bad_mjds.txt","PCA_inputs/nudot_matrix_final.dat",3,235,275,0,True) #DFB -uncut -debug

#do_a_thing = PCA("PCA_inputs/test_pulsar_matrix_1024.txt","PCA_inputs/bad_mjds.txt","PCA_inputs/nudot_matrix_final.dat",3,230,290,55500) # DFB - cut

#AFB
#do_a_thing = PCA("PCA_inputs/B1828-11_AFB_Matrix.txt","PCA_inputs/bad_mjds_AFB.txt","PCA_inputs/nudot_matrix_final.dat",3,80,120) #AFB - uncut


do_a_thing.main()
        
        
        	 


























