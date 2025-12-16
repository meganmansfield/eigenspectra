#Function to fit a map at each wavelength using Eigencurves

#THINGS THAT WILL BE UPDATED IN FUTURE VERSIONS:
#	1. Right now it just loads in a single file, and for the entire pipeline it will load in a file for each wavelength and run the fit for each wavelength
#	2. It will eventually output wavelengths in addition to the spherical harmonic coefficients
#	3. Right now it just selects a number of eigencurves to include. Will eventually optimize this from chi-squared.

#INPUTS:
#	Secondary eclipse light curves at each wavelength (csv for each wavelength)

#OUTPUTS:
#	Coefficients for each of the spherical harmonics
#from lightcurves_sh_starry import sh_lcs
from lightcurves_sh import sh_lcs
from pca_eig import princomp
import numpy as np 
import matplotlib.pyplot as plt 
import emcee
import spiderman as sp
from scipy.optimize import leastsq, minimize
import pdb
import eigenmaps
import os

def mpmodel(p,x,y,z,elc,escore,nparams,degree,ecoeff,wavelength,extent,nonegs):
	#Create model lightcurve and calculate residuals
	model = lightcurve_model(p,elc,escore,nparams)
	if nonegs:
		isnegative,minval = check_negative(degree,p,ecoeff,wavelength,extent)
		if isnegative:
			model = -np.ones(np.shape(y))#*minval
	return np.array(y-model)

def quantile(x, q):
	return np.percentile(x, [100. * qi for qi in q])

def lnprob(theta,x,y,yerr,elc,escore,nparams,degree,ecoeff,wavelength,extent,nonegs=True):
	lp=lnprior(theta)
	if not np.isfinite(lp):
		return -np.inf
	ln_like_val = lnlike(theta,x,y,yerr,elc,escore,nparams,degree,ecoeff,wavelength,extent,nonegs)
	lnprob = ln_like_val + lp
	if not np.isfinite(lnprob):
		return -np.inf 
	else:
		return lnprob

def lnlike(theta,x,y,yerr,elc,escore,nparams,degree,ecoeff,wavelength,extent,nonegs=True):
	#Create model lightcurve and calculate likelihood
	model = lightcurve_model(theta,elc,escore,nparams)
	if nonegs:
		isnegative,minval = check_negative(degree,theta,ecoeff,wavelength,extent)
		if isnegative:
			model = -np.ones(np.shape(y))#*minval
	resid=y-model
	chi2=np.sum((resid/yerr)**2)
	dof=np.shape(y)[0]-1.
	chi2red=chi2/dof
	ln_likelihood=-0.5*(np.sum((resid/yerr)**2 + np.log(2.0*np.pi*(yerr)**2)))
	return ln_likelihood

def lnprior(theta):
	lnpriorprob=0.
	c0 = theta[0]
	fstar = theta[1]
	if fstar<0.:
		lnpriorprob=-np.inf
	elif c0<0.:
		lnpriorprob=-np.inf
	elif c0>1.:
		lnpriorprob=-np.inf
	elif fstar>2.:
		lnpriorprob=-np.inf
	for param in theta[2:]:
		if abs(param)>1.:
			lnpriorprob=-np.inf
	return lnpriorprob

def neg_lnprob(theta,x,y,yerr,elc,escore,nparams,degree,ecoeff,wavelength,extent,nonegs=True):
	return -lnprob(theta,x,y,yerr,elc,escore,nparams,degree,ecoeff,wavelength,extent,nonegs=True)


def lightcurve_model(p,elc,escore,nparams):
	#Compute light curve from model parameters
	model = p[0]*elc[0,:] + p[1]
	for ind in range(2,nparams):
		model = model + p[ind] * escore[ind-2,:]
	return model

def check_negative(degree,fitparams,ecoeff,wavelength,extent):
	#Check whether the resulting map has any negative values at a resolution of 100 x 100 grid points
	#Create spherical harmonic coefficients
	fcoeffs=np.zeros_like(ecoeff)
	for i in np.arange(np.shape(fitparams)[0]-2):
		fcoeffs[:,i] = fitparams[i+2]*ecoeff[:,i]

	sphericalcoeffs=np.zeros(int((degree)**2.))
	for j in range(0,len(fcoeffs)):
		for i in range(1,int((degree)**2.)):
			sphericalcoeffs[i] += fcoeffs.T[j,2*i-1]-fcoeffs.T[j,2*(i-1)]
	sphericalcoeffs[0] = fitparams[0]
	
	#Create 2D map
	londim = 32
	latdim = 16
	
	inputArr=np.zeros([1,sphericalcoeffs.shape[0]+1])
	inputArr[:,0] = wavelength
	inputArr[:,1:] = sphericalcoeffs.transpose()
	wavelengths, lats, lons, maps = eigenmaps.generate_maps(inputArr,N_lon=londim, N_lat=latdim)
	#gotta take extent into account
	visible = np.where((lons[0,:]>extent[0])&(lons[0,:]<extent[1]))[0]
	vismap = maps[0][:,visible]
	negbool = np.any(vismap<0.)
	minval = np.min(vismap)
	return negbool,minval

def makeplot(degree,fitparams,ecoeff,planetparams,eclipsetimes,eclipsefluxes,eclipseerrors):
	#writing this function to easily make a plot to show how well the least squares fit is doing
	fcoeffs=np.zeros_like(ecoeff)
	for i in np.arange(np.shape(fitparams)[0]-2):
		fcoeffs[:,i] = fitparams[i+2]*ecoeff[:,i]

	sphericalcoeffs=np.zeros(int((degree)**2.))
	for j in range(0,len(fcoeffs)):
		for i in range(1,int((degree)**2.)):
			sphericalcoeffs[i] += fcoeffs.T[j,2*i-1]-fcoeffs.T[j,2*(i-1)]
	sphericalcoeffs[0] = fitparams[0]

	t0=planetparams['t0']
	per=planetparams['per']
	a_abs=planetparams['a_abs']
	inc=planetparams['inc']
	ecc=planetparams['ecc']
	planetw=planetparams['w']
	rprs=planetparams['rprs']
	ars=planetparams['ars']
	sparams0=sp.ModelParams(brightness_model='spherical')	#no offset model
	sparams0.nlayers=20

	sparams0.t0=t0				# Central time of PRIMARY transit [days]
	sparams0.per=per			# Period [days]
	sparams0.a_abs=a_abs			# The absolute value of the semi-major axis [AU]
	sparams0.inc=inc			# Inclination [degrees]
	sparams0.ecc=ecc			# Eccentricity
	sparams0.w=planetw			# Argument of periastron
	sparams0.rp=rprs				# Planet to star radius ratio
	sparams0.a=ars				# Semi-major axis scaled by stellar radius
	sparams0.p_u1=0.			# Planetary limb darkening parameter
	sparams0.p_u2=0.			# Planetary limb darkening parameter

	sparams0.degree=degree	#maximum harmonic degree
	sparams0.la0=0.
	sparams0.lo0=0.
	sparams0.sph=list(sphericalcoeffs)

	times=eclipsetimes
	templc=sparams0.lightcurve(times)

	sp.plot_square(sparams0)

	plt.figure()
	plt.plot(times,templc,color='r',zorder=1)
	plt.errorbar(eclipsetimes,eclipsefluxes,yerr=eclipseerrors,linestyle='none',color='k',zorder=0)
	plt.show()

	return templc

def eigencurves(dict,planetparams,homedir,ordmin=3,ordmax=6,eigenmin=2,eigenmax=6,\
			lcName='lcname',plot=False,strict=True,nonegs=True,verbose=False):

	saveDir = homedir + "./data/besteigenlists/" + lcName
	if os.path.exists(saveDir) == False:
		os.makedirs(saveDir)
	
	outputNPZ='{}/prefit.npz'.format(saveDir)
	#unpack data dictionary
	waves=dict['wavelength (um)']
	times=dict['time (days)']
	fluxes=dict['flux (ppm)']	#2D array times, waves
	errors=dict['flux err (ppm)']

	#unpack parameters dictionary
	t0=planetparams['t0']
	per=planetparams['per']
	a_abs=planetparams['a_abs']
	inc=planetparams['inc']
	ecc=planetparams['ecc']
	planetw=planetparams['w']
	rprs=planetparams['rprs']
	ars=planetparams['ars']

	extent=np.zeros(2)
	extent[0]=(np.min(times)-t0)/per*2.*np.pi-np.pi/2.-np.pi
	extent[1]=(np.max(times)-t0)/per*2.*np.pi+np.pi/2.-np.pi
	
	if strict:
		biccut=10.
	else:
		biccut=6.
	
	#This file is going to be a 3D numpy array 
	## The shape is (nsamples,n parameters,n waves)
	## where nsamples is the number of posterior MCMC samples
	## n parameters is the number of parameters
	## and n waves is the number of wavelengths looped over
	
	if np.shape(fluxes)[0]==np.shape(waves)[0]:
		rows=True
	elif np.shape(fluxes)[0]==np.shape(times)[0]:
		rows=False
	else:
		assert (np.shape(fluxes)[0]==np.shape(times)[0]) | (np.shape(fluxes)[0]==np.shape(waves)[0]),"Flux array dimension must match wavelength and time arrays."

	# nParamsUsed = np.zeros(np.shape(waves)[0])

	# eigencurvecoeffList=np.zeros((np.shape(waves)[0],(nsteps-burnin)*nwalkers,int(afew+2))) #FINDME: FIX
	biclist=np.zeros((np.shape(waves)[0],int(ordmax-ordmin+1),int(eigenmax-eigenmin+1)))
	bestcoefflist=np.zeros((np.shape(waves)[0],int(ordmax-ordmin+1),int(eigenmax-eigenmin+1),int(eigenmax+2)))
	degbest=np.zeros(np.shape(waves)[0])
	eigenbest=np.zeros(np.shape(waves)[0])
	for counter in np.arange(np.shape(waves)[0]):
		wavelength=waves[counter] #wavelength this secondary eclipse is for
		if verbose:
			print('Fitting wavelength='+str(wavelength))
		eclipsetimes=times	#in days
		if rows:
			if np.shape(waves)[0]==1:
				eclipsefluxes=fluxes*10.**-6.
				eclipseerrors=errors*10.**-6.
			else:
				eclipsefluxes=fluxes[counter,:]*10.**-6.
				eclipseerrors=errors[counter,:]*10.**-6.
		else:
			if np.shape(waves)[0]==1:
				eclipsefluxes=fluxes*10.**-6.
				eclipseerrors=errors*10.**-6.
			else:
				eclipsefluxes=fluxes[:,counter]*10.**-6.
				eclipseerrors=errors[:,counter]*10.**-6.


		for degnum in np.arange(ordmin,ordmax+1):
			for eigennum in np.arange(eigenmin,eigenmax+1):
				if verbose:
					print('')
					print('')
					print('Fitting for degree='+str(degnum)+', nparams='+str(eigennum))
				#	Calculate spherical harmonic maps using SPIDERMAN
				lc,t = sh_lcs(t0=t0,per=per,a_abs=a_abs,inc=inc,ecc=ecc,w=planetw,\
				rp=rprs,a=ars,ntimes=eclipsetimes,degree=degnum)	#model it for the times of the observations

				# just analyze time around secondary eclipse (from start to end of observations)
				starttime=np.min(eclipsetimes)
				endtime=np.max(eclipsetimes)
				ok=np.where((t>=starttime) & (t<=endtime))
				et = t[ok]
				elc=np.zeros((np.shape(lc)[0],np.shape(ok)[1]))
				for i in np.arange(np.shape(lc)[0]):
					elc[i,:] = lc[i,ok]

				#  PCA
				ecoeff,escore,elatent = princomp(elc[1:,:].T)
				escore=np.real(escore)

				nparams=int(eigennum+2)
				params0=np.zeros(nparams)
				params0[0]=0.0005
				params0[1]=1.
				params0[3]=0.0001

				results = minimize(neg_lnprob, params0, args=(eclipsetimes,eclipsefluxes,\
					eclipseerrors,elc,np.array(escore),nparams,degnum,ecoeff,wavelength,\
					extent,nonegs),method='Nelder-Mead',tol=1e-6,\
				options={'maxiter':None})
				fit_params = results.x
				resid=mpmodel(fit_params,eclipsetimes,eclipsefluxes,eclipseerrors,elc,np.array(escore),nparams,degnum,ecoeff,wavelength,extent,nonegs)
				if verbose:
					print('Negative flux check: '+str(check_negative(degnum,fit_params,ecoeff,wavelength,extent)))
				fit_params_old = params0
				del_params = abs((fit_params_old-fit_params)/(fit_params))
				if verbose:
					print('Delta best fit params from last fit: '+str(np.max(del_params)))
				chi2i=np.sum((resid//eclipseerrors)**2.)
				loglike=-0.5*(np.sum((resid//eclipseerrors)**2 + np.log(2.0*np.pi*(eclipseerrors)**2)))
				bici=-2.*loglike + nparams*np.log(np.shape(eclipseerrors)[0])
				if verbose:
					print('BIC: '+str(bici))
				bicf=bici

				while any(del_params>0.01):
					bici=bicf
					fit_params_old = fit_params
					params0 = fit_params
					results = minimize(neg_lnprob, params0, args=(eclipsetimes,eclipsefluxes,\
						eclipseerrors,elc,np.array(escore),nparams,degnum,ecoeff,wavelength,\
						extent,nonegs),method='Nelder-Mead',tol=1e-6,\
					options={'maxiter':None})
					fit_params = results.x
					resid=mpmodel(fit_params,eclipsetimes,eclipsefluxes,eclipseerrors,elc,np.array(escore),nparams,degnum,ecoeff,wavelength,extent,nonegs)
					if verbose:
						print('Negative flux check: '+str(check_negative(degnum,fit_params,ecoeff,wavelength,extent)))
					del_params = abs((fit_params_old-fit_params)/(fit_params))
					if verbose:
						print('Delta best fit params from last fit: '+str(np.max(del_params)))
					chi2i=np.sum((resid//eclipseerrors)**2.)
					loglike=-0.5*(np.sum((resid//eclipseerrors)**2 + np.log(2.0*np.pi*(eclipseerrors)**2)))
					bicf=-2.*loglike + nparams*np.log(np.shape(eclipseerrors)[0])
					if verbose:
						print('BIC: '+str(bicf))
				
				bicf=-2.*loglike + nparams*np.log(np.shape(eclipseerrors)[0])
				bestcoeffs=fit_params
				print(bestcoeffs)
				#print(bici,bicf,bici-bicf)
				biclist[counter,degnum-ordmin,eigennum-eigenmin]=bicf
				bestcoefflist[counter,degnum-ordmin,eigennum-eigenmin,:nparams]=bestcoeffs

				#make an eclipse lightcurve for one wavelength
				# plotyvals=bestcoeffs[0]*elc[0,:]+bestcoeffs[1]
				# for i in np.arange(int(np.shape(bestcoeffs)[0]-2)):
				# 	plotyvals+=bestcoeffs[i+2]*escore[i,:]
				# pdb.set_trace()

				if plot:
					# translate coefficients
					fcoeffbest=np.zeros_like(ecoeff)
					for i in np.arange(np.shape(bestcoeffs)[0]-2):
						fcoeffbest[:,i] = bestcoeffs[i+2]*ecoeff[:,i]

					# how to go from coefficients to best fit map
					spheresbest=np.zeros(int((degnum)**2.))
					for j in range(0,len(fcoeffbest)):
						for i in range(1,int((degnum)**2.)):
							spheresbest[i] += fcoeffbest.T[j,2*i-1]-fcoeffbest.T[j,2*(i-1)]
					spheresbest[0] = bestcoeffs[0]#c0_best

					params0=sp.ModelParams(brightness_model='spherical')	#no offset model
					params0.nlayers=20

					params0.t0=t0				# Central time of PRIMARY transit [days]
					params0.per=per			# Period [days]
					params0.a_abs=a_abs			# The absolute value of the semi-major axis [AU]
					params0.inc=inc			# Inclination [degrees]
					params0.ecc=ecc			# Eccentricity
					params0.w=planetw			# Argument of periastron
					params0.rp=rprs				# Planet to star radius ratio
					params0.a=ars				# Semi-major axis scaled by stellar radius
					params0.p_u1=0.			# Planetary limb darkening parameter
					params0.p_u2=0.			# Planetary limb darkening parameter

					params0.degree=degnum	#maximum harmonic degree
					params0.la0=0.
					params0.lo0=0.
					params0.sph=list(spheresbest)

					times=eclipsetimes
					templc=params0.lightcurve(times)

					sp.plot_square(params0)

					plt.figure()
					plt.plot(times,templc,color='r',zorder=1)
					plt.errorbar(eclipsetimes,eclipsefluxes,yerr=eclipseerrors,linestyle='none',color='k',zorder=0)
					plt.show()

				
		# bicbest=np.argmin(biclist[counter,:,:])
		bestindex=np.unravel_index(biclist[counter].argmin(),biclist[counter].shape)
		bicbest=biclist[counter,bestindex[0],bestindex[1]]
		# pdb.set_trace()
		if verbose:
			print('Best degree='+str(ordmin+bestindex[0])+'; Best num eigencurves='+str(eigenmin+bestindex[1]))
		degbest[counter]=ordmin+bestindex[0]
		eigenbest[counter]=eigenmin+bestindex[1]

		plt.figure()
		plt.imshow(biclist[counter]-np.min(biclist[counter]),extent=[eigenmin-0.5,eigenmax+0.5,ordmax+0.5,ordmin-0.5],vmax=100)
		plt.colorbar()
		plt.title('$\Delta$BIC, wave='+str(wavelength))
		plt.xlabel('Number of Eigencurves')
		plt.ylabel('Degree')
		plt.savefig(saveDir+'/delta_waveind_'+str(counter))
		plt.show()

		# newfile=open(saveDir+'/besteigenlist_'+str(counter)+'.txt')
		# print(bestcoefflist[counter,bestindex[0],bestindex[1],:nparams])
		# newfile.close()
		
	finaldict={'wavelength (um)':waves,'BIC':biclist,'bestcoeffs':bestcoefflist,\
	'best fit degree':degbest,'best fit num eigencurves':eigenbest,'degmin':ordmin,\
	'eigenmin':eigenmin}
	# finaldict={'wavelength (um)':waves,'N Params Used':nParamsUsed,\
	# 			'eigencurve coefficients':eigencurvecoeffList,'BIC':biclist}

	np.savez(outputNPZ,finaldict)
	# newfile=open(saveDir+'/best_degree.txt','w')
	# print(degbest,file=newfile)
	# newfile.close()
	# newfile=open(saveDir+'/best_num_eigen.txt','w')
	# print(eigenbest,file=newfile)
	# newfile.close()


	return finaldict