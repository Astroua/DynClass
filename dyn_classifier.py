import numpy as np
from matplotlib import pyplot as plt
from astropy.table import Table

def dyn_classifier(Vc, verbose = True):

	# Load dynamical classifier from 238 CALIFA galaxies
	# from Kalinova et al. 2017b
	dynclass = np.load('./data_input/dyn_classes.npz')
	# ... dynamical radius
	rads = dynclass['rads']
	# ... eigenvectors
	us = dynclass['eigenvectors']
	ls = dynclass['eigenvalues']
	# ... mean Vc
	MeanVal = dynclass['Vc']
	# ... prototype curves
	cvc_SR = dynclass['cvc_SR'] - MeanVal
	cvc_FL = dynclass['cvc_FL'] - MeanVal
	cvc_RP = dynclass['cvc_RP'] - MeanVal
	cvc_SP = dynclass['cvc_SP'] - MeanVal

	pcvcs = np.zeros([4,len(cvc_SP)], dtype=np.float64)
	pcvcs[0,:] = cvc_SR
	pcvcs[1,:] = cvc_FL
	pcvcs[2,:] = cvc_RP
	pcvcs[3,:] = cvc_SP

	cols = ['k','b','g','r']
	centroids = np.zeros([4,2])

	# Decompose prototypes on the PC base
	for i in [0,1,2,3]:
		centroids[i,0]=np.sum(pcvcs[i,:]*us[0,:])/np.sqrt(ls[0])
	   	centroids[i,1]=np.sum(pcvcs[i,:]*us[1,:])/np.sqrt(ls[1])


	# Load user CVC
	Vc = Vc - MeanVal

	pc1 = np.sum(Vc*us[0,:])/np.sqrt(ls[0])
	pc2 = np.sum(Vc*us[1,:])/np.sqrt(ls[1])
	pcs = np.array([pc1,pc2])

	# Measure the distance to the centroids

	dists = np.zeros(4)
	for i in [0,1,2,3]:
		dists[i] = np.linalg.norm(pcs-centroids[i,:])

	mdist = np.argmin(dists)
	percs = 100-dists/np.sum(dists)*100
	percs = percs.astype(np.int)
	tags = np.array(['SR', 'FL', 'RP', 'SP'])
	names = np.array(['slow rising', 'flat', 'round peaked', 'sharp peaked'])

	for i in np.argsort(percs):
		tag = tags[i]
		name = names[i]

		if verbose:
			print "Your CVC is "+str(percs[i])+"%  closer to "+tag
	
	if verbose:
		print "Your CVC is "+names[mdist]+" ("+tags[mdist]+")!"
		
	ucol = cols[mdist]
	res = tags[mdist]


	# Plot the result
	if verbose:

		fig = plt.figure(figsize=(14,7))
		ax1 = fig.add_subplot(121)
		ax2 = fig.add_subplot(122)

		# Decompose prototypes on the PC base
		for i in [0,1,2,3]:
			ax1.plot(rads, pcvcs[i,:]+MeanVal, color = cols[i], lw = 3, ls='-')
			ax2.scatter([centroids[i][0]],[centroids[i][1]],c = cols[i], marker = '*', s = 500, edgecolor = 'face')

		ax1.plot(rads, Vc+MeanVal, color = ucol, lw = 2, ls='--', label = 'User CVC')
		ax1.legend(numpoints = 1, loc = 4)
		ax2.scatter([pc1],[pc2],c = 'w', marker = 'o', s = 250, edgecolor = ucol, lw = 2, label = 'User CVC')
		ax2.legend(numpoints = 1, loc = 4)

		ax1.set_xlabel('Radius/Reff')
		ax1.set_ylabel('Vc [km/s]')

		ax2.set_xlabel('PC1')
		ax2.set_ylabel('PC2')
		ax2.plot([-1.8,3.],[0.,0.],ls='--',color='k')
		ax2.plot([0.,0.],[-5.,2.],ls='--',color='k')
		ax2.set_xlim([-1.8,3.])
		ax2.set_ylim([-5.,2.])

	return res


#%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

def tutorial():

	data = Table.read('./data_input/example.txt',format='ascii')
	Vcirc = data['Vcirc/[km/s]'].data

	dclass = dyn_classifier(Vcirc,verbose=True)



#%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

if __name__ == '__main__':
	tutorial()

