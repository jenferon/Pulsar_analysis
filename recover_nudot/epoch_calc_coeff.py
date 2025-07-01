import numpy as np
import matplotlib.pyplot as plt
import argparse

class epochs_calculation():

	def __init__(self, widths, jump):
		if isinstance(widths,str):
			self.widths = np.loadtxt(widths)
			arg = np.argmin(np.abs(self.widths[:,0]-58000))
			self.widths=self.widths[:arg, :]
		elif isinstance(widths,np.ndarray):
			self.widths = widths
		
		self.jump = jump
		
	def print_output(self,number, mjd, jump):
		print("GLEP_"+str(number)+"	"+str(mjd))
		print("GLF1_"+str(number)+"	"+str(jump))
	
	def calc(self,widths):
		avg_width = np.mean(widths[:,1])

		low_to_high = np.array(())
		high_to_low = np.array(())
		x = 1

		for i in range(len(widths[:,0])-1):#-1 to stop index error
			
			if widths[i,1]<avg_width and widths[i+1,1]>avg_width:
				low_to_high = np.append(low_to_high, (widths[i,0]+widths[i+1,0])/2)
				self.print_output(x,(widths[i,0]+widths[i+1,0])/2,"-"+str(self.jump))
				x += 1
				
			elif widths[i,1]>avg_width and widths[i+1,1]<avg_width:
				high_to_low = np.append(high_to_low, (widths[i,0]+widths[i+1,0])/2)
				self.print_output(x,(widths[i,0]+widths[i+1,0])/2,self.jump)
				x += 1
		return [avg_width, low_to_high, high_to_low]
		
	def plot(self,widths, avg_width, low_to_high, high_to_low):
		
			
		plt.plot(widths[:,0], widths[:,1], label ="coeff")
		plt.axhline((avg_width), xmin=0.01, xmax=1, linestyle='--', color='k', linewidth=1, label = "mean coeff")
		for i in range(len(low_to_high)):
			plt.axvline((low_to_high[i]), ymin=0.01, ymax=1, linestyle='--', color='blue', linewidth=1, label="low to high", alpha =0.5)
		
		for i in range(len(high_to_low)):
			plt.axvline((high_to_low[i]), ymin=0.01, ymax=1, linestyle='-.', color='red', linewidth=1, label="high to low", alpha = 0.5)
		plt.xlabel("MJD")
		plt.ylabel("coeff")
		plt.savefig("calculated_transitions.png")
		plt.show()

		
	def run(self):
		[avg_width, low_to_high, high_to_low] = self.calc(self.widths)
		self.plot(self.widths, avg_width, low_to_high, high_to_low)



def main():

	parser = argparse.ArgumentParser(description='Plots variability map above nudot graph')
	parser.add_argument('-w','--widths', help='file containing mjd, w_10, w_10 error ', required=True)
	parser.add_argument('-j','--jump', help =  'optional jump size: default = 2e-15',required = False, default = 2e-15)
	args = parser.parse_args()
	
	DoAThing = epochs_calculation(args.widths,args.jump)
	DoAThing.run()
   
if __name__ == '__main__':
    main()


