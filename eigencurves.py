#Function to fit a map at each wavelength using Eigencurves

#INPUTS:
#	Secondary eclipse light curves at each wavelength (csv for each wavelength)

#OUTPUTS:
#	Coefficients for each of the spherical harmonics

from lightcurves_sh import sh_lcs
from pca_eig import princomp
import numpy as np 
import matplotlib.pyplot as plt 
import emcee
import spiderman as sp
from scipy.optimize import leastsq, minimize
import pdb
from matplotlib import rc
import eigenmaps
import multiprocessing as mp

def mpmodel(p,x,y,z,elc,escore,nparams,degree,ecoeff,wavelength,extent,nonegs):
	#Create model lightcurve and calculate residuals
	model = lightcurve_model(p,elc,escore,nparams)
	if nonegs:
		isnegative,minval = check_negative(degree,p,ecoeff,wavelength,extent)
		if isnegative:
			model = -np.ones(np.shape(y))
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
			model = -np.ones(np.shape(y))
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
	#only consider visible regions of the planet
	visible = np.where((lons[0,:]>extent[0])&(lons[0,:]<extent[1]))[0]
	vismap = maps[0][:,visible]
	negbool = np.any(vismap<0.)
	minval = np.min(vismap)
	return negbool,minval

def makeplot(degree,fitparams,ecoeff,planetparams,eclipsetimes,eclipsefluxes,eclipseerrors):
	#make a plot to show how well the least squares fit is doing
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

def eigencurves(dict,planetparams,plot=False,degree=3,afew=5,burnin=100,nsteps=1000,prefit=False,nonegs=True,verbose=False):
	
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
	
	nwalkers=100 #number of walkers, hard coded in for now
	biccut=6. #BIC difference needed to prefer a more complex fit
	
	# Create variables to hold output
	if not prefit:
		alltheoutput=np.zeros(((nsteps-burnin)*nwalkers,int((degree)**2.),np.shape(waves)[0]))
		bestfitoutput=np.zeros((int((degree)**2.),np.shape(waves)[0]))
	
	if np.shape(fluxes)[0]==np.shape(waves)[0]:
		rows=True
	elif np.shape(fluxes)[0]==np.shape(times)[0]:
		rows=False
	else:
		assert (np.shape(fluxes)[0]==np.shape(times)[0]) | (np.shape(fluxes)[0]==np.shape(waves)[0]),"Flux array dimension must match wavelength and time arrays."

	nParamsUsed = np.zeros(len(waves))
	biclist=np.zeros(len(waves))
	ecoeffList, escoreList,elatentList = [], [], []
	fullchainarray=[]
	eigencurvecoeffList = []
	elclist=[]
	for counter in np.arange(np.shape(waves)[0]): #loop through each wavelength
		wavelength=waves[counter] #wavelength this secondary eclipse is for
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

		#	Calculate spherical harmonic maps using SPIDERMAN
		lc,t = sh_lcs(t0=t0,per=per,a_abs=a_abs,inc=inc,ecc=ecc,w=planetw,rp=rprs,a=ars,ntimes=eclipsetimes,degree=degree)

		# just analyze from start to end of observations
		starttime=np.min(eclipsetimes)
		endtime=np.max(eclipsetimes)
		ok=np.where((t>=starttime) & (t<=endtime))
		et = t[ok]
		elc=np.zeros((np.shape(lc)[0],np.shape(ok)[1]))
		for i in np.arange(np.shape(lc)[0]):
			elc[i,:] = lc[i,ok]

		# Run the PCA (Rauscher et al. 2018)
		ecoeff,escore,elatent = princomp(elc[1:,:].T)
		escore=np.real(escore)

		if isinstance(afew,list):
			print('Gave an array of afew values')
			if not np.shape(waves)[0]==np.shape(afew)[0]:
				assert (np.shape(waves)[0]==np.shape(afew)[0]), "Array of afew values must be the same length as the number of wavelength bins, which is"+str(np.shape(waves)[0])
			nparams=int(afew[counter]+2)
		else:
			if not isinstance(afew,int):
				assert isinstance(afew,int), "afew must be an integer >=1!"
			elif afew>=16:
				print('Performing fit for best number of eigencurves to use.')
				delbic=20.
				nparams=4
				params0=np.zeros(nparams)
				params0[0]=0.0005
				params0[1]=1.
				if verbose:
					print(check_negative(degree,params0,ecoeff,wavelength,extent))
				mpfit=leastsq(mpmodel,params0,args=(eclipsetimes,eclipsefluxes,eclipseerrors,elc,np.array(escore),nparams,degree,ecoeff,wavelength,extent,nonegs))
				resid=mpmodel(mpfit[0],eclipsetimes,eclipsefluxes,eclipseerrors,elc,np.array(escore),nparams,degree,ecoeff,wavelength,extent,nonegs)
				if plot:
					throwaway=makeplot(degree,mpfit[0],ecoeff,planetparams,eclipsetimes,eclipsefluxes,eclipseerrors)
				if verbose:
					print(check_negative(degree,mpfit[0],ecoeff,wavelength,extent))
				chi2i=np.sum((resid//eclipseerrors)**2.)
				loglike=-0.5*(np.sum((resid//eclipseerrors)**2 + np.log(2.0*np.pi*(eclipseerrors)**2)))
				bici=-2.*loglike + nparams*np.log(np.shape(eclipseerrors)[0])
				tempparams=mpfit[0]

				while delbic>biccut:
					nparams+=1
					if nparams==15:
						params0=np.zeros(nparams)
						params0[0]=0.0005
						params0[1]=1.
						if verbose:
							print(check_negative(degree,params0,ecoeff,wavelength,extent))
						mpfit=leastsq(mpmodel,params0,args=(eclipsetimes,eclipsefluxes,eclipseerrors,elc,np.array(escore),nparams,degree,ecoeff,wavelength,extent,nonegs))
						chi2f=np.sum((mpmodel(mpfit[0],eclipsetimes,eclipsefluxes,eclipseerrors,elc,np.array(escore),nparams,degree,ecoeff,wavelength,extent,nonegs)//eclipseerrors)**2.)
						delbic=5.

					else:
						params0=np.zeros(nparams)
						params0[0]=0.0005
						params0[1]=1.
						if verbose:
							print(check_negative(degree,params0,ecoeff,wavelength,extent))
						mpfit=leastsq(mpmodel,params0,args=(eclipsetimes,eclipsefluxes,eclipseerrors,elc,np.array(escore),nparams,degree,ecoeff,wavelength,extent,nonegs))
						resid=mpmodel(mpfit[0],eclipsetimes,eclipsefluxes,eclipseerrors,elc,np.array(escore),nparams,degree,ecoeff,wavelength,extent,nonegs)
						if plot:
							throwaway=makeplot(degree,mpfit[0],ecoeff,planetparams,eclipsetimes,eclipsefluxes,eclipseerrors)
						if verbose:
							print(check_negative(degree,mpfit[0],ecoeff,wavelength,extent))
						chi2f=np.sum((resid//eclipseerrors)**2.)
						loglike=-0.5*(np.sum((resid//eclipseerrors)**2 + np.log(2.0*np.pi*(eclipseerrors)**2)))
						bicf=-2.*loglike + nparams*np.log(np.shape(eclipseerrors)[0])
						delbic=bici-bicf
						if verbose:
							print('Current number of eigencurves:',nparams-2)
							print('Delta BIC:',bici-bicf)
							print('Best-fit params:',mpfit[0])
							print('Difference from n-1 iteration:',mpfit[0][:-1]-tempparams)
						chi2i=chi2f
						bici=bicf
						tempparams=mpfit[0]
				print('BIC criterion says the best number of eigencurves to use is '+str(nparams-3))
				nparams-=1
			
			elif ((afew<16)&(afew>=1)):
				nparams=int(afew+2)

			else:
				assert afew>1 ,"afew must be an integer 1<=afew<=15!"
		
		params0=np.zeros(nparams)
		params0[0]=0.0005
		params0[1]=1.

		if verbose:
			print(check_negative(degree,params0,ecoeff,wavelength,extent))
		mpfit=leastsq(mpmodel,params0,args=(eclipsetimes,eclipsefluxes,eclipseerrors,elc,np.array(escore),nparams,degree,ecoeff,wavelength,extent,nonegs))
		if plot:
			throwaway=makeplot(degree,mpfit[0],ecoeff,planetparams,eclipsetimes,eclipsefluxes,eclipseerrors)
		if verbose:
			print(check_negative(degree,mpfit[0],ecoeff,wavelength,extent))

		#format parameters for mcmc fit
		theta=mpfit[0]
		ndim=np.shape(theta)[0]	#set number of dimensions
		ndim=np.shape(theta)[0]	#set number of dimensions
		stepsize=0.001*theta
		sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(eclipsetimes,eclipsefluxes,eclipseerrors,elc,escore,nparams,degree,ecoeff,wavelength,extent,nonegs),pool=mp.Pool(5))
		pos = [theta + stepsize*np.random.randn(ndim) for i in range(nwalkers)]
		print("Running MCMC at {} um".format(waves[counter]))

		bestfit=np.zeros(ndim+1)
		bestfit[0]=10.**8

		sampler.run_mcmc(pos,nsteps,progress=True)

		try:
			print(sampler.get_autocorr_time())
		except:
			print('WARNING: Could not estimate autocorrelation time. We strongly recommend running a longer chain!')
			
		samples=sampler.get_chain(discard=burnin,flat=True)
		for i in np.arange(ndim):
			bestfit[i+1]=quantile(samples[:,i],[0.5])[0]
		
		resid=mpmodel(bestfit[1:],eclipsetimes,eclipsefluxes,eclipseerrors,elc,np.array(escore),nparams,degree,ecoeff,wavelength,extent,nonegs)
		chi2f=np.sum((resid//eclipseerrors)**2.)
		bestfit[0]=chi2f
		loglike=-0.5*(np.sum((resid//eclipseerrors)**2 + np.log(2.0*np.pi*(eclipseerrors)**2)))
		bicf=-2.*loglike + nparams*np.log(np.shape(eclipseerrors)[0])
		biclist[counter]=bicf

		fullchain=sampler.chain
		fullchainarray.append(fullchain)

		bestcoeffs=bestfit[1:]

		#make best fit eclipse lightcurve
		plotyvals=bestcoeffs[0]*elc[0,:]+bestcoeffs[1]
		for i in np.arange(int(np.shape(bestcoeffs)[0]-2)):
			plotyvals+=bestcoeffs[i+2]*escore[i,:]

		# translate coefficients
		fcoeffbest=np.zeros_like(ecoeff)
		for i in np.arange(np.shape(bestcoeffs)[0]-2):
			fcoeffbest[:,i] = bestcoeffs[i+2]*ecoeff[:,i]

		# go from coefficients to best fit map
		spheresbest=np.zeros(int((degree)**2.))
		for j in range(0,len(fcoeffbest)):
			for i in range(1,int((degree)**2.)):
				spheresbest[i] += fcoeffbest.T[j,2*i-1]-fcoeffbest.T[j,2*(i-1)]
		spheresbest[0] = bestcoeffs[0]
		bestfitoutput[:,counter]=spheresbest
		
		for sampnum in np.arange(np.shape(samples)[0]):
			fcoeff=np.zeros_like(ecoeff)
			for i in np.arange(np.shape(samples)[1]-2):
				fcoeff[:,i] = samples[sampnum,i+2]*ecoeff[:,i]

			# go from coefficients to best fit map
			spheres=np.zeros(int((degree)**2.))
			for j in range(0,len(fcoeff)):
				for i in range(1,int((degree)**2.)):
					spheres[i] += fcoeff.T[j,2*i-1]-fcoeff.T[j,2*(i-1)]
			spheres[0] = samples[sampnum,0]	
			
			alltheoutput[sampnum,:,counter]=spheres

		if plot:
			finlc=makeplot(degree,bestfit[1:],ecoeff,planetparams,eclipsetimes,eclipsefluxes,eclipseerrors)

		nParamsUsed[counter]=nparams
		ecoeffList.append(ecoeff)
		escoreList.append(escore)
		eigencurvecoeffList.append(samples)
		elatentList.append(elatent)
		elclist.append(elc)
		
	
	finaldict={'wavelength (um)':waves,'spherical coefficients':alltheoutput,\
	'best fit spherical coefficients':bestfitoutput,'N Params Used':nParamsUsed,\
	'ecoeffList': ecoeffList,'escoreList': escoreList,'elc': elclist,\
	'eigencurve coefficients':eigencurvecoeffList,'BIC':biclist,'elatentList':elatentList,\
	'fullchainarray':fullchainarray}

	return finaldict
