# Import modules

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor as gpr
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

import logging
import os
import sys


class GpPredict(object):

    def __init__(self, data=None, template=None, off_pulse_std=None, start=None, end=None, allbins=1024, debug_level=logging.DEBUG):

        logging.basicConfig(level=debug_level)

        logging.debug("GpPredict instantiated")

        self.data = data
        self.template = template
        self.nbins = None
        self.nprofiles = None
        self.off_pulse_std = off_pulse_std
        self.interval = None
        self.start = start
        self.end = end
        self.allbins = allbins

    def form_residuals(self):

        logging.debug("Entering gppredict.form_residuals")
        
        self.nbins = self.data.shape[0]
        self.nprofiles = self.data.shape[1]

        # Form array of profile residuals
        residuals = np.zeros((self.nbins, self.nprofiles))

        # Go through each profile and subtract the template
        for i in range(self.nprofiles):
            logging.debug("Subtracting profile {} from template".format(i))
            residuals[:,i] = self.data[:,i] - self.template

        logging.debug("Leaving gppredict.form_residuals")
        return residuals

    def predict(self, mjds=None, residuals=None, interval=1, length_scale=165, length_scale_bounds=(30, 300)):

        logging.debug("Entering gppredict.predict")

        self.interval = interval

        # Create directory to store plots for each bin
        if not os.path.isdir("posteriors"):
            os.mkdir("posteriors")
        else:
            logging.warning("Posteriors directory already exists")

        # Define inference points for GP
        mjdinfer = np.arange(mjds[0], mjds[-1], self.interval)

        max_diff = np.amax(residuals)
        min_diff = np.min(residuals)
        
        # Define covariance function
        kernel1 = 1.0 * Matern(length_scale=length_scale, length_scale_bounds=length_scale_bounds, nu=1.5) 
        kernel2 = WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
        kernel = kernel1 + kernel2

        # Create empty arrays for inferred residuals, variances and limits
        inferred_array = np.zeros((self.nbins, mjdinfer.shape[0]))
        inferred_var= np.zeros((self.nbins, mjdinfer.shape[0]))
        lower_limit = np.zeros(mjdinfer.shape[0])
        upper_limit = np.zeros(mjdinfer.shape[0])

        # Go through each bins and train gp on all epochs, 
        # then predict the values at our training epochs
        for i in range(self.nbins):
            training_resids = residuals[i,:]
            logging.info("=================Training GP on bin {} of {}==================".format(i+1, self.nbins))
            inferred_array[i,:], inferred_var[i,:] = self._gpfit(mjds, training_resids, mjdinfer, kernel)

            upper_limit[:] = inferred_array[i,:] + 2 * np.sqrt(inferred_var[i,:])
            lower_limit[:] = inferred_array[i,:] - 2 * np.sqrt(inferred_var[i,:])

            # Plot the posteriors
            plt.plot(mjds, residuals[i,:], 'r.')
            plt.plot(mjdinfer, inferred_array[i,:], 'k-')
            plt.fill_between(mjdinfer, lower_limit, upper_limit, color='b', alpha=0.2)
            plt.title("Bin {}".format(i+1))
            plt.xlabel("Training epoch [MJD]", fontsize=15)
            plt.ylabel("Training residuals [s]", fontsize=15)
            bin_number = str(i+1).zfill(4)
            outfile = "posteriors/bin_" + str(bin_number) + ".png"
            plt.savefig(outfile, format='png', dpi=400)
            plt.clf()

        outputfile = "inferred_array.dat"
        np.savetxt(outputfile, inferred_array)

        self.makemap(inferred_array, mjdinfer, mjds, self.off_pulse_std)

        logging.debug("Leaving gppredict.predict")
        return mjdinfer, inferred_array

    def makemap(self, inferred_array, mjdinfer, mjds, off_pulse_std):
        logging.debug("Entered makemap method")
       
        inferred_array = inferred_array / off_pulse_std
 
        # Set up figure panel and set dimensions
        logging.debug("Setting up map parameters")
        fig=plt.figure()
        fig.set_size_inches(16,10)

        xbins = inferred_array.shape[1]
        logging.info("xbins {}".format(xbins))

        ybins = inferred_array.shape[0]
        logging.info("ybins {}".format(ybins))

        maxdifference = np.amax(inferred_array)
        mindifference = np.amin(inferred_array)
        limitdifference = np.max((maxdifference, np.abs(mindifference)))

        # Plot profile residuals
        logging.debug("Plotting map") 
        plt.imshow(inferred_array*1.0,
            aspect = "auto",
            cmap = "RdBu_r",
            vmin = -limitdifference,
            vmax = limitdifference,
            interpolation='gaussian')

        # Overlay observation epochs on variability map
        logging.debug("Adding observation epochs") 
        for i in range(0, len(mjds)):
            plt.axvline((mjds[i]-mjdinfer[0])/float(self.interval), ymin=0.01, ymax=0.05, linestyle='solid', color='k', linewidth=1)

        plt.ylabel("Pulse phase", fontsize=20)
        plt.xlabel("Modified Julian Day", fontsize=20)

        # Set up locations at which we want to place x-axis tick labels
        logging.debug("Handling x tick labels") 
        xlocs = np.arange(xbins, step=1000/(1 * float(self.interval)))
        xticklabels = []
        for i in xlocs:
            xticklabels.append(int(mjdinfer[int(i)]))
        plt.xticks(xlocs, xticklabels, rotation="horizontal")
       
        # Add line showing profile peak location 
        logging.debug("Adding profile peak") 
        peakline = self.allbins/4-self.start
        plt.axhline(peakline, color='k', linestyle='solid')

        # Set up locations at which we want to place y-axis tick labels
        logging.debug("Handling y tick labels") 
        yaxis = []
        yaxis.append(np.linspace(0, (self.end - self.start)/self.allbins, self.end - self.start))
        yaxis = yaxis[0]
        ylocs = np.linspace(0, ybins, 10)
        yticklabels = []
        for i in ylocs[:-1]:
            yticklabels.append(round(yaxis[int(i)], 2))
        plt.yticks(ylocs, yticklabels)
 
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.ylim([max(ylocs), min(ylocs)])

        logging.debug("Adding colorbar") 
        cbaxes = fig.add_axes([0.35, 0.94, 0.55, 0.01])
        cb3 = plt.colorbar(cax=cbaxes, orientation="horizontal")
        cb3.update_ticks()
        cb3.ax.tick_params(labelsize=24)

        logging.debug("Saving plot") 
        plt.savefig("variability_map.pdf", format='pdf', dpi=400)
        plt.close()
        logging.debug("Leaving makemap method")


    @staticmethod
    def _gpfit(training_mjds, training_residuals, test_mjds, kernel):

        logging.debug("Entered private method gppredict._gpfit")

        # Reshape input domain to comply with sklearn's requirements
        training_mjds = np.reshape(training_mjds, (len(training_mjds), 1))

        # Train GP on input data
        gp_model = gpr(kernel=kernel, alpha=0.0,
                n_restarts_optimizer=10).fit(training_mjds, training_residuals)

        maxlik = gp_model.log_marginal_likelihood(gp_model.kernel_.theta)
        logging.info("Max log_likelihood: {}".format(maxlik))
        opt_hyp = gp_model.kernel_
        logging.info("Optimised parameters:\n{}".format(opt_hyp))

        test_mjds = np.reshape(test_mjds, len(test_mjds), 0)
        res_mean, res_cov = gp_model.predict(test_mjds[:,np.newaxis], return_cov=True)

        res_var = 1.0 * np.diag(res_cov)
        logging.debug("Leaving private method gppredict._gpfit")
        return np.array(res_mean.T), np.array(res_var.T)

