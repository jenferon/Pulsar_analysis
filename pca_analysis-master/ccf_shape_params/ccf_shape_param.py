#compare ccf measurements for shape parameters of pulse profile with pca output
import numpy as np
import matplotlib.pyplot as plt
from ccf_interpolate import ccf

class CCF(object):
	def __init__(self, double_str, meshed_str, single_str, fix_relphase_width_str, fix_relphase_amp_str, single_nudot_str, double_nudot_str, meshed_nudot_str, relphase_nudot_str, profile_mjds_str):
	
		#load matricies
		single_nudot_matrix = np.loadtxt(single_nudot_str)
		double_nudot_matrix = np.loadtxt(double_nudot_str)
		meshed_nudot_matrix = np.loadtxt(meshed_nudot_str)
		relphase_nudot_matrix = np.loadtxt(relphase_nudot_str)
		single_matrix = np.loadtxt(single_str)
		double_matrix = np.loadtxt(double_str)
		meshed_matrix = np.loadtxt(meshed_str)
		fix_relphase_width_matrix = np.loadtxt(fix_relphase_width_str)
		fix_relphase_amp_matrix = np.loadtxt(fix_relphase_amp_str)

		#initilise class
		self.single_nudot = single_nudot_matrix[:,1]
		self.single_nudot_mjds = single_nudot_matrix[:,0] 
		self.single_nudot_err = single_nudot_matrix[:,2]
		self.double_nudot = double_nudot_matrix[:,1]
		self.double_nudot_mjds = double_nudot_matrix[:,0] 
		self.double_nudot_err = double_nudot_matrix[:,2] 
		self.meshed_nudot = meshed_nudot_matrix[:,1]
		self.meshed_nudot_mjds = meshed_nudot_matrix[:,0] 
		self.meshed_nudot_err = meshed_nudot_matrix[:,2] 
		self.relphase_nudot = relphase_nudot_matrix[:,1]
		self.relphase_nudot_mjds = relphase_nudot_matrix[:,0] 
		self.relphase_nudot_err = relphase_nudot_matrix[:,2] 
		self.single = single_matrix[:,1] 
		self.single_mjds = single_matrix[:,0] 
		self.single_err = single_matrix[:,2] 
		self.double = double_matrix[:,1] 
		self.double_mjds = double_matrix[:,0] 
		self.double_err = double_matrix[:,2] 
		self.meshed = meshed_matrix[:,1] 
		self.meshed_mjds = meshed_matrix[:,0] 
		self.meshed_err = meshed_matrix[:,2]
		self.fix_relphase_width = fix_relphase_width_matrix[:,1] 
		self.fix_relphase_width_mjds = fix_relphase_width_matrix[:,0] 
		self.fix_relphase_width_err = fix_relphase_width_matrix[:,2]
		self.fix_relphase_amp = fix_relphase_amp_matrix[:,1] 
		self.fix_relphase_amp_mjds = fix_relphase_amp_matrix[:,0] 
		self.fix_relphase_amp_err = fix_relphase_amp_matrix[:,2]
		self.profile_mjds = np.loadtxt(profile_mjds_str)
	

	def plotter_err(self, lags, cc_coeff, err, name, title):
	#plot the ccf against lags
		plt.errorbar(lags, cc_coeff, err, color = "k", ecolor= "red", elinewidth= 0.5)
		plt.xlabel('lags (days)')
		plt.ylabel('cc_coeff')
		plt.title(title)
		plt.savefig(name)
		plt.show()	
		
	def plotter(self, lags, cc_coeff, name, title):
	#plot the ccf against lags
		plt.plot(lags, cc_coeff)
		plt.xlabel('lags (days)')
		plt.ylabel('cc_coeff')
		plt.title(title)
		plt.savefig(name)
		plt.show()
	
	def sub_plot_err(self, x, y, err, x_label, y_label, name, letter):
	#subplot for dataframe
		
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
				ax.set_ylabel(y_label[i], fontsize=7)
				ax.get_yaxis().set_label_coords(-0.1,0.5)
				ax.text(-1000,0.7, letter[i])
				if i != np.shape(y)[0]-1:
					ax.xaxis.set_ticklabels([])
		except IndexError:
				fig, ax = plt.subplots()
				ax.errorbar(x,y,err,color = "k", ecolor= "red", elinewidth= 0.5)
				ax.set_ylabel(y_label[0], fontsize=5)
		ax.set_xlabel(x_label)
		plt.subplots_adjust(hspace=.0)
		plt.savefig(name)
		plt.show()	
	
	def mjd_checker(self, mjds, name):
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
		plt.savefig("profile_time_step_hist_" + name + ".png")
		plt.show()
	
	def plot_checker(self):
		plt.errorbar(self.single_mjds, self.single, self.single_err)
		plt.xlabel('mjd')
		plt.ylabel('w10')
		plt.title("Single W10")
		plt.show()
		
		plt.errorbar(self.double_mjds, self.double, self.double_err)
		plt.xlabel('mjd')
		plt.ylabel('w10')
		plt.title("Double W10")
		plt.show()
		
		plt.errorbar(self.meshed_mjds, self.meshed, self.meshed_err)
		plt.xlabel('mjd')
		plt.ylabel('w10')
		plt.title("Meshed W10")
		plt.show()
		
		plt.errorbar(self.fix_relphase_width_mjds, self.fix_relphase_width, self.fix_relphase_width_err)
		plt.xlabel('mjd')
		plt.ylabel('w10')
		plt.title("Fix Relphase W10")
		plt.show()
		
		plt.errorbar(self.fix_relphase_amp_mjds, self.fix_relphase_amp, self.fix_relphase_amp_err)
		plt.xlabel('mjd')
		plt.ylabel('Amp ratio')
		plt.title("Fix Relphase Amp ratio")
		plt.show()
		
		#check nudots
		plt.errorbar(self.single_nudot_mjds, self.single_nudot, self.single_nudot_err)
		plt.xlabel('mjd')
		plt.ylabel('nudot')
		plt.title("Single")
		plt.show()
		
		plt.errorbar(self.double_nudot_mjds, self.double_nudot, self.double_nudot_err)
		plt.xlabel('mjd')
		plt.ylabel('nudot')
		plt.title("Double")
		plt.show()
		
		plt.errorbar(self.meshed_nudot_mjds, self.meshed_nudot, self.meshed_nudot_err)
		plt.xlabel('mjd')
		plt.ylabel('nudot')
		plt.title("Meshed")
		plt.show()
		
		plt.errorbar(self.relphase_nudot_mjds, self.relphase_nudot, self.relphase_nudot_err)
		plt.xlabel('mjd')
		plt.ylabel('nudot')
		plt.title("Relphase")
		plt.show()
		
		
	def plot_cc_coeff(self, nudot, nudot_mjds, profile, profile_mjds, name, title, n):
		
		#organises inputs to for use in cross_correlate and plots output on subplots
		rel_cadence = 5
		lag_lim = 1000
		lags = np.arange(-lag_lim,lag_lim,rel_cadence)
		#lags = np.arange(-(np.around(np.max(profile_mjds)-np.min(profile_mjds))),np.around(np.max(profile_mjds)-np.min(profile_mjds)),1)/n

		cc_coeff = cc_coeff = ccf(nudot, profile ,nudot_mjds, profile_mjds, method = "gaussian",lags=lags, min_weighting=0, max_gap = 40).results()
		
		cc_coeff_array = cc_coeff[:,1]
		cc_coeff_err = cc_coeff[:,2]
		plot_lags = cc_coeff[:,0]
		
		self.mjd_checker(profile_mjds, name)
		self.weighting_checker(cc_coeff[:,3], name)
		
		#self.plotter_err(plot_lags, cc_coeff_array, cc_coeff_err, name, title)
		
		return cc_coeff
		
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
                y_label =[]
                letter = []
                
                single = self.plot_cc_coeff(self.single_nudot, self.single_nudot_mjds, self.single, self.single_mjds, "single_ccf.png", "Single CCF", 4.5)
                ccf_zero_lag = single[np.where(single[:,0]==0)[0],1:3]
                cc_coeff_array = single[:,1]
                cc_coeff_err = single[:,2]
                y_label.append(r"DCF($\dot{\nu}$,$w_{10}$)")
                letter.append("(i)")
                
                double = self.plot_cc_coeff(self.double_nudot, self.double_nudot_mjds, self.double, self.double_mjds, "double_ccf.png", "Double CCF", 4.5)
                ccf_zero_lag = np.vstack((ccf_zero_lag,double[np.where(double[:,0]==0)[0],1:3]))
                cc_coeff_array = np.vstack((cc_coeff_array, double[:,1]))
                cc_coeff_err = np.vstack((cc_coeff_err, double[:,2]))
                y_label.append(r"DCF($\dot{\nu}$,$w_{10}$)")
                letter.append("(ii)")
                
                meshed = self.plot_cc_coeff(self.meshed_nudot, self.meshed_nudot_mjds, self.meshed, self.meshed_mjds, "meshed_ccf.png", "Meshed CCF", 4.5)
                ccf_zero_lag = np.vstack((ccf_zero_lag,meshed[np.where(meshed[:,0]==0)[0],1:3]))
                cc_coeff_array = np.vstack((cc_coeff_array, meshed[:,1]))
                cc_coeff_err = np.vstack((cc_coeff_err, meshed[:,2]))
                y_label.append(r"DCF($\dot{\nu}$,$w_{10}$)")
                letter.append("(iii)")
                
                relphase_widths = self.plot_cc_coeff(self.relphase_nudot, self.relphase_nudot_mjds, self.fix_relphase_width, self.fix_relphase_width_mjds, "relphase_width_ccf.png", "Relphase Widths CCF", 4.5)
                ccf_zero_lag = np.vstack((ccf_zero_lag,relphase_widths[np.where(relphase_widths[:,0]==0)[0],1:3]))
                cc_coeff_array = np.vstack((cc_coeff_array, relphase_widths[:,1]))
                cc_coeff_err = np.vstack((cc_coeff_err, relphase_widths[:,2]))
                y_label.append(r"DCF($\dot{\nu}$,$w_{10}$)")
                letter.append("(iv)")
                
                relphase_amps = self.plot_cc_coeff(self.relphase_nudot, self.relphase_nudot_mjds, self.fix_relphase_amp, self.fix_relphase_amp_mjds, "relphase_amps_ccf.png", "Relphase Amps CCF", 4.5)
                ccf_zero_lag = np.vstack((ccf_zero_lag,relphase_amps[np.where(relphase_amps[:,0]==0)[0],1:3]))
                cc_coeff_array = np.vstack((cc_coeff_array, relphase_amps[:,1]))
                cc_coeff_err = np.vstack((cc_coeff_err, relphase_amps[:,2]))
                y_label.append(r"DCF($\dot{\nu}$,$A_{pre}/A_{main})$")
                letter.append("(v)")

                np.savetxt("ccf_zero_lag_shape_params.txt", ccf_zero_lag)
                
                self.sub_plot_err(single[:,0], cc_coeff_array, cc_coeff_err, "lag (days)", y_label, "cross_correlation_shape_params.png", letter)

do_a_thing = CCF("double_widths.dat", "meshed_widths.dat", "single_widths.dat", "fix_relphase_widths.dat", "fix_relphase_amps.dat", "single_nudot.dat", "double_nudot.dat", "meshed_nudot.dat", "relphase_nudot.dat", "profile_mjds.dat")

do_a_thing.main()		
