#!/usr/bin/env python

from __future__ import print_function
import logging
import argparse
import os
import sys
import traceback
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import matplotlib.pyplot as plt
from scipy.linalg import lapack as lp
from scipy import spatial


# pylint: disable=C0116
# pylint: disable=C0103
# pylint: disable=C0301
# pylint: disable=W1202

class calcNudot():


    def __init__(self, ephemeris, residuals, lengthscale, glitch, mjd_nudot_err, cadence=None):


        self.ephemeris = ephemeris # Pulsar ephemeris
        self.residuals = residuals # Timing residuals
        self.cadence = cadence # Cadence
        self.lengthscale = lengthscale[0]
        if isinstance(glitch, str):
            self.glitch = np.loadtxt(glitch)
        elif isinstance(glitch,np.ndarray):
            self.glitch = glitch	
        mjd_nudot_err = np.loadtxt(mjd_nudot_err)

        try:
            self.lengthscale2 = lengthscale[1]
            if self.lengthscale2 == self.lengthscale:
                raise Exception("Lengthscales should not be the same initial value")
        except IndexError:
            self.lengthscale2 = False
        self.pepoch = None

        logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(message)s]', level=logging.INFO)
        


        self.uncut_mjd = mjd_nudot_err[:,0]
        self.uncut_nudot = mjd_nudot_err[:,1]
        self.uncut_err = mjd_nudot_err[:,2]
        
    def _check_file(self, this_file):
        """
        Checks our input files exist and
        exits if not
        """
        outcome = True
        if not os.path.isfile(this_file):
            logging.info("{} not found".format(this_file))
            outcome = False
        return outcome

    def _read_eph(self, eph):
        """
        Reads ephemeris and extracts
        period and period derivative
        """
        with open(eph, 'r') as this_eph:
            for line in this_eph.readlines():
                fields = line.split()
                if fields[0] == "F0":
                    period = 1 / float(fields[1])
                if fields[0] == "F1":
                    fdot = float(fields[1])
                if fields[0] == "PEPOCH":
                    self.pepoch = float(fields[1])
            logging.info("Period is {}".format(period))
            return period, fdot

    def get_opt_pars(self, model):
        """
        Extracts optimised hyperparameters from GP model
        """
        if not self.lengthscale2:
            variance = np.exp(model.kernel_.theta[0])
            lengthscale = np.exp(model.kernel_.theta[1])
            noisevar = np.exp(model.kernel_.theta[2])
            return variance, lengthscale, noisevar
        else:
            variance = np.exp(model.kernel_.theta[0])
            lengthscale = np.exp(model.kernel_.theta[1])
            variance2 = np.exp(model.kernel_.theta[2])
            lengthscale2 = np.exp(model.kernel_.theta[3])
            noisevar = np.exp(model.kernel_.theta[4])
            return variance, lengthscale, variance2, lengthscale2, noisevar 
        return None

    def make_cov(self, mjd):
        """
        Constructs covariance matrix using optimised hyperparameters
        """
        if not self.lengthscale2:
            K = []
            for i in range(0, len(mjd)):
                this_cov = []
                for j in range(0, len(mjd)):
                    this_cov.append(self.get_cov(i,
                        j,
                        mjd[i],
                        mjd[j],
                        self.opt_pars[0],
                        self.opt_pars[1],
                        self.opt_pars[2]))
                K.append(this_cov)
        else:
            K = []
            for i in range(0, len(mjd)):
                this_cov = []
                for j in range(0, len(mjd)):
                    this_cov.append(self.get_cov_len2(i,
                        j,
                        mjd[i],
                        mjd[j],
                        self.opt_pars[0],
                        self.opt_pars[1],
                        self.opt_pars[2],
                        self.opt_pars[3],
                        self.opt_pars[4]))
                K.append(this_cov)

        K = np.asarray(K)
        return K

    @staticmethod
    def get_cov(i, j, resi, resj, var, lengthscale, noisevar):
        """
        Computes covariance elements for single lengthscale kernel
        """
        if i != j:
            cov = var * np.exp(-0.5 * (resi - resj)**2.0 * (1.0 / lengthscale**2.0))
        else:
            cov = var * np.exp(-0.5 * (resi - resj)**2.0 * (1.0 / lengthscale**2.0)) + noisevar

        return cov[0]

    @staticmethod
    def get_cov_len2(i, j, resi, resj, var, lengthscale, var2, lengthscale2, noisevar):
        """
        Computes covariance elements from double kernel.
        """
        if i != j:
            cov = var * np.exp(-0.5 * (resi - resj)**2.0 * (1.0 / lengthscale**2.0)) + var2 * np.exp(-0.5 * (resi - resj)**2.0 * (1.0 / lengthscale2**2.0))
        else:
            cov = var * np.exp(-0.5 * (resi - resj)**2.0 * (1.0 / lengthscale**2.0)) + var2 * np.exp(-0.5 * (resi - resj)**2.0 * (1.0 / lengthscale2**2.0)) + noisevar

        return cov[0]

    def pdinv(self, A, *args):
        """
        :param A: A DxD pd numpy array

        :rval Ai: the inverse of A
        :rtype Ai: np.ndarray
        :rval L: the Cholesky decomposition of A
        :rtype L: np.ndarray
        :rval Li: the Cholesky decomposition of Ai
        :rtype Li: np.ndarray
        :rval logdet: the log of the determinant of A
        :rtype logdet: float64

        """
        L = self.jitchol(A, *args)
        logdet = 2.*np.sum(np.log(np.diag(L)))
        Li = lp.dtrtri(L)
        Ai, _ = lp.dpotri(L, lower=1)
        # Ai = np.tril(Ai) + np.tril(Ai,-1).T
        self.symmetrify(Ai)

        return Ai, L, Li, logdet

    def jitchol(self, A, maxtries=5):
        A = np.ascontiguousarray(A)
        L, info = lp.dpotrf(A, lower=1)
        if info == 0:
            return L
        else:
            diagA = np.diag(A)
            if np.any(diagA <= 0.):
                raise linalg.LinAlgError("not pd: non-positive diagonal elements")
            jitter = diagA.mean() * 1e-6
            num_tries = 1
            while num_tries <= maxtries and np.isfinite(jitter):
                try:
                    L = linalg.cholesky(A + np.eye(A.shape[0]) * jitter, lower=True)
                    return L
                except:
                    jitter *= 10
                finally:
                    num_tries += 1
            raise linalg.LinAlgError("not positive definite, even with jitter.")
        try: raise
        except:
            logging.warning('\n'.join(['Added jitter of {:.10e}'.format(jitter),
                '  in '+traceback.format_list(traceback.extract_stack(limit=3)[-2:-1])[0][2:]]))
        return L

    def symmetrify(self, A, upper=False):
        """
        Take the square matrix A and make it symmetrical by copting elements from
        the lower half to the upper

        works IN PLACE.

        note: tries to use cython, falls back to a slower numpy version
        """
        self.symmetrify_numpy(A, upper)


    def symmetrify_numpy(self, A, upper=False):
        triu = np.triu_indices_from(A,k=1)
        if upper:
            A.T[triu] = A[triu]
        else:
            A[triu] = A.T[triu]

    def DKD(self, X1, X2, theta):

        X1, X2 = np.matrix(X1), np.matrix(X2) # ensure both sets of inputs are matrices

        D2 = spatial.distance.cdist(X1, X2, 'sqeuclidean') # calculate squared Euclidean distance
        D1 = np.zeros((X1.shape[0],X2.shape[0]))
        K = theta[0] * np.exp(- D2 / (2*(theta[1]**2))) * ( theta[1]**2 - D2) / theta[1]**4

        return np.matrix(K)

    def write_data(self, mjd, nudot, err):
        outfile = "mjd_nudot_err.dat"
        if os.path.isfile(outfile):
            logging.info("{} already exists. Overwriting".format(outfile))
            os.remove(outfile)
        dat = np.array([mjd, nudot, err]).T
        np.savetxt(outfile, dat, delimiter = ' ')

    def plot_model_residuals(self, mjd, res, reserr, res_mean, sig):

        fig, ax = plt.subplots(nrows=2, ncols=1)
        plt.subplot(211)

        # Plot model
        plt.plot(mjd, res_mean, 'k-', zorder=2)
        # Plot data
        plt.errorbar(mjd, res, yerr=reserr, marker='.', color='r', ecolor='r', linestyle='None', zorder=1, alpha=0.5)
        upper_bound = res_mean + (2 * sig)
        lower_bound = res_mean - (2 * sig)
        flat_mjds = []
        for i in range(0, len(mjd)):
            flat_mjds.append((mjd[i][0]))
        plt.fill_between(flat_mjds, upper_bound, lower_bound, color = 'k', alpha = 0.2)
        plt.ylabel("Residual [s]", fontsize=15)
        plt.grid()
        frame = plt.gca()
        frame.axes.xaxis.set_ticklabels([])

        plt.subplot(212)
        plt.errorbar(self.pepoch + np.asarray(mjd), res - res_mean, yerr=reserr, color='k', marker='.', ecolor='k', linestyle='None')
        plt.xlabel("Modified Julian Day", fontsize=12)
        plt.ylabel("Data - model [s]", fontsize=12)
        plt.grid()
        plt.tight_layout()
        plt.savefig("Residuals.pdf", format='pdf')
        plt.show()


    def run(self):

        # Check files exists and exit if one or both is missing
        logging.info("Checking files exists")
        if self._check_file(self.residuals) and \
           self._check_file(self.ephemeris):
            pass
        else:
            raise FileNotFoundError("One or more files not supplied")

        # Extract period and frequency derivative from ephemeris
        logging.info("Reading parameters from ephemeris")
        self.period, self.f1 = self._read_eph(self.ephemeris)

        # Load in residuals
        logging.info("Loading residuals")
        mjdin, res, err = np.loadtxt(self.residuals, unpack=True, usecols=[0,1,2])

        # Reshape mjd array according to scikit learn's requirements
        mjd = np.reshape(mjdin, (len(mjdin), 1))

        # Define kernel function
        # This is the radial basis function combined with a white kernel
        K_RBF = 1.0 * RBF(length_scale=self.lengthscale)
        if self.lengthscale2:
            K_RBF += 1.0 * RBF(length_scale=self.lengthscale2)
        K_WHITE = WhiteKernel(noise_level=1, noise_level_bounds=(1e-8, 1e+1))
        K_RBF += K_WHITE

        # Set up GP model
        logging.info("Setting up GP model")
        gp_model = GPR(
            kernel=K_RBF,
            alpha=0.0,
            n_restarts_optimizer = 10)
        # Get initial hyperparameters
        init_hyp = gp_model.kernel
        logging.info("Initial hyperparameters: {}".format(init_hyp))

        # Train GP model on input data
        logging.info("Training GP on residuals...")
        gp_model.fit(mjd, res)

        # Get optmised hyperparameters
        opt_hyp = gp_model.kernel_
        self.opt_pars = self.get_opt_pars(gp_model)
        if not self.lengthscale2:
            logging.info("Signal variance: {}".format(self.opt_pars[0]))
            logging.info("Lengthscale: {}".format(self.opt_pars[1]))
            logging.info("Noise Variance: {}".format(self.opt_pars[2]))
        else:
            logging.info("First signal variance: {}".format(self.opt_pars[0]))
            logging.info("First lengthscale: {}".format(self.opt_pars[1]))
            logging.info("Second signal variance: {}".format(self.opt_pars[2]))
            logging.info("Second lengthscale: {}".format(self.opt_pars[3]))
            logging.info("Noise Variance: {}".format(self.opt_pars[4]))

        # Get max log likelihood for optmised hyperparameters
        maxlik = gp_model.log_marginal_likelihood(gp_model.kernel_.theta)
        logging.info("Max log-likelihood is {}".format(maxlik))

        # Set the mjds at which we want to infer the residuals
        # For now this is just the mjds at which we observed
        if self.cadence:
            mjdinfer = np.arange(mjd[0][0], mjd[-1][0], float(self.cadence))
            mjdinfer = np.reshape(mjdinfer, (len(mjdinfer), 1))
        else:
            mjdinfer = mjd

        # Infer residual at our mjds (test epochs) based on trained GP
        res_mean, res_cov = gp_model.predict(mjdinfer, return_cov=True)

        # Get variance at each of our test epochs
        # This is the square root of the diagonals of our covariance matrix
        sig = 1.0 * np.sqrt(np.diag(res_cov))

        # Plot model residuals so we can check for remaining structure.
        self.plot_model_residuals(mjd, res, err, res_mean, sig) 

        # Reformulate arrays for next steps of processing
        res_mean = np.asarray(res_mean)
        sig = np.asarray(sig)

        # To Do: Make residuals and residuals of residuals plot

        logging.info("Constructing covariance matrix")
        self.K = self.make_cov(mjd)

        logging.info("Inverting covariance matrix")
        K1invOut = self.pdinv(np.matrix(self.K))
        K1inv = K1invOut[1]

        logging.info("Setting prediction epochs")
        XTRAINING = mjd
        XPREDICT = mjd
        YTRAINING = np.matrix(np.array(res.flatten())).T

        logging.info("Calculating nudot")
        logging.info("Kernel 1")
        covFunc = self.DKD
        par = np.zeros(2)
        par[0] = self.opt_pars[0]
        par[1] = self.opt_pars[1]
        K_prime = covFunc(XPREDICT, XTRAINING, par)
        K_prime_p = 3*par[0]/par[1]**4
        if self.lengthscale2:
            logging.info("Kernel 2")
            par2 = np.zeros(2)
            par2[0] = self.opt_pars[2]
            par2[1] = self.opt_pars[3]
            K_prime += covFunc(XPREDICT, XTRAINING, par2)
            K_prime_p += 3*par2[0]/par2[1]**4

        KiKx, _ = lp.dpotrs(K1inv, np.asfortranarray(K_prime.T), lower = 1)

        nudot_raw = np.array(self.f1  + np.dot(KiKx.T, YTRAINING)/self.period/(86400)**2)
        nudot_err_raw = np.array(np.sqrt(K_prime_p - np.sum(np.multiply(KiKx, K_prime.T),0).T)/(86400)**2)
        nudot = [ i[0]/1e-15 for i in nudot_raw]
        nudot_err = [ i[0]/1e-15 for i in nudot_err_raw]
        mjds = [ i[0] for i in mjdinfer]
        self.write_data(self.pepoch + np.asarray(mjds), nudot, nudot_err)

	
        diff_min = np.abs(self.uncut_mjd - np.min(self.pepoch + np.asarray(mjds)))
        first_mjd_arg = np.argmin(diff_min)
			
        diff_max = np.abs(self.uncut_mjd - np.max(self.pepoch + np.asarray(mjds)))
        last_mjd_arg = np.argmin(diff_max)
	
		#slices the data to the min and the max mjd
        cut_mjd = self.uncut_mjd[first_mjd_arg:(last_mjd_arg+1)] 
        cut_nudot = self.uncut_nudot[first_mjd_arg:(last_mjd_arg+1)]
        cut_err = self.uncut_err[first_mjd_arg:(last_mjd_arg+1)]
	
	#plot 3 graphs on one axis        
        actual_mjd = self.pepoch + np.asarray(mjds)
        plt.errorbar(self.pepoch + np.asarray(mjds), nudot, yerr=nudot_err, color='k', ecolor='k', marker='.', linestyle='solid',label ="simulation output")
        plt.xlabel("Modified Julian Day", fontsize=15)
        plt.ylabel("Frequency derivative [1e-15 Hz/s]", fontsize=15)
        #plt.tight_layout()
        '''
        t = np.linspace(actual_mjd[0], actual_mjd[-1], actual_mjd[-1]-actual_mjd[0], endpoint=False)
        y = np.full((len(t), 1), -365.6)
        n=0
        for i in range(0,len(y)):
            y[i] = y[i] + ((-1)**(n+1))*(1)
            if n == 28:
               break
            if np.rint(t[i]) == self.glitch[n]:
                n = n+1
                
            else:
                continue
        
        plt.plot(t,y,label ="simulation input")  
		'''
        for i in range(0, len(self.glitch)):
            plt.axvline((self.glitch[i]), ymin=0.01, ymax=0.05, linestyle='solid', color='k', linewidth=1)
        plt.errorbar(cut_mjd, cut_nudot, yerr=cut_err, label ="Real data")
        plt.legend()
        plt.savefig("fake_nudot.pdf", format='pdf', dpi=400)
        plt.show()

def main():

    parser = argparse.ArgumentParser(description='Calculates the evolution of the spin-down rate from pulsar residuals using Gaussian Process Regression')
    parser.add_argument('-e','--ephemeris', help='Pular Ephemeris', required=True)
    parser.add_argument('-r','--residuals', help='File of fake residuals [mjd, res, err]', required=True)
    parser.add_argument('-c','--cadence', help='Cadence of test epochs (Not yet implemented)', required=False)
    parser.add_argument('-l', '--lengthscales', help="List of lengthscales (up to 2)", type=float, nargs='+', default=[500.0])
    parser.add_argument('-g','--glitch_epochs', help="List of epochs where a glitch happens",required=False, default = np.array(()) )
    parser.add_argument('-n','--mjd_nudot_err', help="list of actual residuals", required=True)
    args = parser.parse_args()
    args.lengthscales = np.asarray(args.lengthscales)
	

    measure = calcNudot(args.ephemeris, args.residuals, args.lengthscales, args.glitch_epochs, args.mjd_nudot_err, args.cadence)

    measure.run()

if __name__ == '__main__':
    main()
