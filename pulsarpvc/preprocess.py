# Import modules

import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt
import logging
import shutil

import os
import sys


class Preprocess(object):

    def __init__(self, rawdata=None, chosen_level=logging.INFO):

        logging.basicConfig(level=chosen_level)

        logging.debug("Instantiated Preprocess")

        if self.verify_file(rawdata):
            self.rawdata = np.loadtxt(rawdata)
            self.verify_data(self.rawdata)

    def filter(self, badmjds_file):

        logging.debug("Entering preprocess.filter")

        bad_mjds = np.loadtxt(badmjds_file)
        bad_mjds = np.atleast_1d(bad_mjds)

        index = []
        for mjd in bad_mjds:
            for i in range(self.rawdata.shape[1]):
                if abs(self.rawdata[self.rawdata.shape[0]-1,i]-mjd) < 0.01:
                    index.append(i)

        filtered_data = np.delete(self.rawdata, index, 1)
        logging.debug("Leaving preprocess.filter")
        return filtered_data

    def stripinput(self, data):

        logging.debug("Entering preprocess.stripinput")

        mjds = data[data.shape[0]-1,:]
        tobs = data[data.shape[0]-2,:]
        data = data[0:data.shape[0]-2,:]

        bins = data.shape[0]
        logging.debug("Leaving preprocess.stripinput")

        return data, mjds, tobs, bins 

    def debase(self, data, outlier_threshold=2):

        logging.debug("Entering preprocess.debase")
        nbins = data.shape[0]
        nprofiles = data.shape[1]

        debased = data
        lowestrms = np.zeros(nprofiles)
        lowestmean = np.zeros(nprofiles)
        peak = np.zeros(nprofiles)

        for i in range(nprofiles):
            rms = np.zeros(8)
            mean = np.zeros(8)
            section = int(nbins / 8)
            for j in range(8):
                mean[j] = np.mean(data[j*section:(j+1)*section,i])
                rms[j] = np.std(data[j*section:(j+1)*section,i])
            lowestrms[i] = np.min(rms)
            peak[i] = np.max(data[:,i])
            baseindex = np.argmin(rms)
            baseline = mean[baseindex]
            lowestmean[i] = baseline
            debased[:,i] = data[:,i] - baseline

        medianrms = np.median(lowestrms)
        medianpeak = np.median(peak)

        outlierindex = []
        inlierindex = []

        for i in range(nprofiles):
            if lowestrms[i]/np.max(data[:,i]) > outlier_threshold * medianrms/medianpeak:
                outlierindex.append(i)
            else:
                inlierindex.append(i)

        ou = np.array(outlierindex)
        inl = np.array(inlierindex)

        removedprofiles = np.delete(debased, inl, 1)
        debasedoutlierremoved = np.delete(debased, ou, 1)
        rms_removed = np.delete(lowestrms, ou)
        logging.info("Removed outliers {}".format(ou.shape[0]))

        logging.debug("Leaving preprocess.debase")
        return debasedoutlierremoved, removedprofiles, rms_removed, ou, inl
            

    def find_brightest(self, data, rms_data):
        
        logging.debug("Entering preprocess.find_brightest")
        snr = np.zeros(rms_data.shape[0])

        for i in range(data.shape[1]):
            snr[i] = np.max(data[:,i])/rms_data[i]
        
        brightest_index = np.argmax(snr)

        brightest_peak = np.max(data[:,brightest_index])
        brightest_rms = rms_data[brightest_index]

        if brightest_peak / brightest_rms < 20:
            logging.warn("Brightest profile is less than 20 sigma. Consider resampling")
    
        logging.debug("Leaving preprocess.find_brightest")
        return brightest_index

    def resample(self, data, factor=8):

        logging.debug("Entering preprocess.resample")
        nbins = data.shape[0]
        resampled = ss.resample(data, int(nbins/factor))
        logging.debug("Resampled data by factor of {}".format(factor))

        return resampled

    def align(self, data, brightest_index):
        logging.debug("Entering preprocess.align")

        nbins = data.shape[0]
        nprofiles = data.shape[1]

        brightest = data[:,brightest_index]

        # Roll template to 0.25 phase and plot it
        peakbin = np.argmax(brightest)
        fixedlag = int(nbins/4)-peakbin
        
        rotated_brightest = np.roll(brightest, fixedlag)
        brightest = rotated_brightest
        logging.debug("Brightest profile rolled by {} bins to 0.25 phase position".format(fixedlag))

        # Now we want to cross correlate each profile with the 
        # rotated brightest profile and rotate them all to the same
        # phase (0.25 of a turn). 
        aligned = np.zeros((nbins,nprofiles))
        for i in range(nprofiles):
            logging.debug("Cross correlating profile {} with brightest".format(i+1))
            xcorr = np.correlate(brightest, data[:,i], "full")
            lag = np.argmax(xcorr)
            logging.debug("Rolling profile {} by {} bins".format(i+1, lag))
            aligned[:,i] = np.roll(data[:,i],lag)

        # Repeat with improved template which is formed from 
        # the median value of each bin over all observations
        template = np.median(aligned, 1)
        peakbin = np.argmax(template)
        fixedlag = int(nbins/4)-peakbin

        # We double up the profile
        double = np.zeros(2*nbins)
        for i in range(nprofiles):
            # Populate the first nbins bins with the profile
            double[0:nbins] = data[:,i]
            # Repeat for the secon half of the profile
            # The profile is now doubled
            double[nbins:2*nbins] = data[:,i]
            xcorr = np.correlate(template,double,"full")
            #xcorr = np.correlate(template,data[:,i],"full")
            lag = np.argmax(xcorr) + fixedlag
            logging.debug("Rolling profile {} by {} bins".format(i+1, lag))
            aligned[:,i] = np.roll(data[:,i],lag)
            # Form template (again) for return
            final_template = np.median(aligned, 1)

        logging.debug("Leaving preprocess.align")
        return np.array(aligned), np.array(brightest), np.array(final_template)

    def get_on_pulse_data(self, data, template, mjds, start, end):

        logging.debug("Entering preprocess.get_on_pulse_data")
        nbins = data.shape[0]
        nprofiles = data.shape[1]

        on_pulse_region = list(range(start, end))
        on_pulse_data = data[on_pulse_region,]

        normed = self.normalise(data, on_pulse_data)

        for i in range(data.shape[0]):
            template[i]=np.median(normed[i,:])

        template_peak = np.max(template)
        template = template/template_peak
        normed = normed/template_peak

        # Next we calculate the difference between 
        # each bin value of each profile and the same
        # bin in the template profile
        sum_diff = np.zeros((normed.shape[1]))

        for i in range(normed.shape[1]):
            for k in range(normed.shape[0]):
                sum_diff[i]+=abs(normed[k,i]-template[k])

        if not os.path.isdir("flagged_profiles"):
            os.mkdir("flagged_profiles")
        else:
            logging.warning("Flagged profiles directory already exists - removing")
            shutil.rmtree("flagged_profiles")
            os.mkdir("flagged_profiles")
            logging.info("Flagged directory removed and re-created")
            

        flagged = 0
        for j in range(normed.shape[1]):
            if sum_diff[j] > 1.5 * np.mean(sum_diff[:]):
                logging.info("Flagging MJD {} as unusual".format(mjds[j]))
                flagged+=1
                outname = "flagged_profiles/" + str(mjds[j]) + ".png"
                plt.plot(normed[:,j], 'b-')
                plt.plot(template[:], 'r-')
                plt.xlabel("Phase bins", fontsize=15)
                plt.ylabel("Normalised power", fontsize=15)
                plt.text(nbins/3, 0.9,'Deviation from Median Profile: {0:.4f}' .format(sum_diff[j],fontsize=8))
                plt.text(nbins/3, 0.8,'Mean of Profile Deviation: {0:.4f}' .format(np.mean(sum_diff[:]),fontsize=8))
                plt.text(nbins/3, 0.7,'STDev of Profile Deviation: {0:.4f}' .format(np.std(sum_diff[:]),fontsize=8))
                plt.title(mjds[j])
                plt.savefig(outname)
                plt.close()

        logging.info("{} profiles flagged as unusual".format(flagged))
        logging.info("{} profiles out of {} seem fine".format(normed.shape[1] - flagged, normed.shape[1]))

        output_array = normed[start:end,:]
        plt.imshow(output_array, aspect='auto')
        plt.xlabel("Observation index", fontsize=15)
        plt.ylabel("Pulse phase", fontsize=15)
        plt.colorbar(orientation="horizontal")
        plt.savefig("zoomed_pulse_stack.png")
        plt.close()

        # Write out debased, aligned, normalised and zoomed array to file
        #output_array_file = "zoomed_" + str(start) + "_" + str(end) + ".txt"
        #if os.path.isfile(output_array_file):
        #    logging.warning("{} already exists".format(output_array_file))
        #np.savetxt(output_array_file, output_array)

        if not os.path.isdir("good_profiles"):
            os.mkdir("good_profiles")
        else:
            logging.warning("Good profiles directory already exists")
            shutil.rmtree("good_profiles")
            os.mkdir("good_profiles")
            logging.info("good_profiles deleted and recreated")

        # Plot good profiles 
        for i in range(nprofiles):
            outname = "good_profiles/" + str(mjds[i]) + ".png"
            logging.debug("Plotting {}".format(outname))
            plt.plot(output_array[:,i], 'k-')
            plt.plot(template[start:end], 'r-')
            plt.xlabel("Phase bins", fontsize=15)
            plt.ylabel("Normalised power", fontsize=15)
            plt.title(mjds[i])
            plt.tight_layout()
            plt.savefig(outname)
            plt.close()

        # Get median value of each profile (all bins)
        bin_med = []
        bin_std = []
        for i in range(nbins):
            bin_med.append(np.median(normed[i,:]))
            bin_std.append(np.std(normed[i,:]))

        # Find the off-pulse standard deviation for use in next step

        off_pulse_data = np.delete(normed, on_pulse_region, 0)
        std_off_pulse_data = np.std(off_pulse_data)

        # Pass debased, aligned, normalised and zoomed array to caller and exit
        logging.debug("Leaving preprocess.get_on_pulse_data")
        return output_array, bin_med, bin_std, std_off_pulse_data, template[start:end]

    @staticmethod
    def normalise(data, on_pulse):
        logging.debug("Entered Normaliser method")

        nprofiles = data.shape[1]

        for i in range(nprofiles):
            data[:,i] = data[:,i] / np.mean(on_pulse[:,i])

        return data
  
    @staticmethod
    def plot(data, mjds, outdir):

        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        else:
            logging.warning("Directory {} already exists".format(outdir))
            shutil.rmtree(outdir)
            os.mkdir(outdir)
            logging.info("Removed and recreated {}".format(outdir))

        nprofiles = data.shape[1]

        for i in range(nprofiles):

            logging.debug("Plotting profile from mjd {} into {}".format(mjds[i], outdir))

            filename = outdir + "/" + str(mjds[i]) + ".png"
            plt.plot(data[:,i], 'k-')
            plt.xlabel("Phase bins", fontsize=15)
            plt.ylabel("Power [arb' units]", fontsize=15)
            plt.title(mjds[i])
            plt.tight_layout()
            plt.savefig(filename)
            plt.close()

    @staticmethod
    def plot_single(data, title=None):

        filename = str(title) + ".png"

        plt.plot(data, 'k-')
        plt.xlabel("Phase bins", fontsize=15)
        plt.ylabel("Power [arb' units]", fontsize=15)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    @staticmethod
    def verify_file(datafile):
        
        # Test the user has supplied a file
        if datafile == None:
            logging.info("No input file supplied")
            return False
        
        # Test that the file supplied exists
        if not os.path.isfile(datafile): 
            logging.info("Cannot find {}".format(datafile))
            return False

        return True

    @staticmethod
    def verify_data(data):

        # How many observations have we found?
        logging.info("Found {} observations".format(data.shape[1]))

        # Test that each observations
        # has the same number of bins
        for i in range(1, data.shape[1]):
            if len(data[:,i]) != len(data[:,i-1]):
                logging.error("Cannot proceed. Observations have different numbers of bins")
                sys.exit(9)
        logging.info("Observations have {} bins".format(len(data[:,1])-2))

