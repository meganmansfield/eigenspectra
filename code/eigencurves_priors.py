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
from scipy.optimize import leastsq
# import multiprocessing as mp
import pdb
import eigenmaps

def mpmodel(p,x,y,z,elc,escore,nparams,degree,ecoeff,wavelength,extent,nonegs):
	#Create model lightcurve and calculate residuals
	model = lightcurve_model(p,elc,escore,nparams)
	if nonegs:
		isnegative,minval = check_negative(degree,p,ecoeff,wavelength,extent)
		if isnegative:
			model = np.ones(np.shape(y))#*minval
	return np.array(y-model)

def lnprob(theta,x,y,yerr,elc,escore,nparams,degree,ecoeff,wavelength,extent,params0,nonegs=True):
	lp=lnprior(theta,params0)
	return lp+lnlike(theta,x,y,yerr,elc,escore,nparams,degree,ecoeff,wavelength,extent,nonegs)

def lnlike(theta,x,y,yerr,elc,escore,nparams,degree,ecoeff,wavelength,extent,nonegs=True):
	#Create model lightcurve and calculate likelihood
	model = lightcurve_model(theta,elc,escore,nparams)
	if nonegs:
		isnegative,minval = check_negative(degree,theta,ecoeff,wavelength,extent)
		if isnegative:
			model = np.ones(np.shape(y))#*minval
	resid=y-model
	chi2=np.sum((resid/yerr)**2)
	dof=np.shape(y)[0]-1.
	chi2red=chi2/dof
	ln_likelihood=-0.5*(np.sum((resid/yerr)**2 + np.log(2.0*np.pi*(yerr)**2)))
	return ln_likelihood

def lnprior(theta,params0):
	lnpriorprob=0.
	c0 = theta[0]
	fstar = theta[1]
	if fstar<0.5:#0.
		lnpriorprob=-np.inf
	# elif c0<0.:
	# 	lnpriorprob=-np.inf
	# elif c0>1.:
	# 	lnpriorprob=-np.inf
	elif fstar>1.5:#1.
		lnpriorprob=-np.inf
	means=params0
	# errors=params0*0.01
	# for index in np.arange(np.shape(theta)[0]):
	# 	lnpriorprob -= (0.5*(np.sum(((theta[index]-means[index])/errors[index])**2
    #                          + np.log(2.0*np.pi*(errors[index])**2))))
	for index in np.arange(np.shape(theta)[0]):
		if abs(theta[index])>100.*abs(means[index]):
			lnpriorprob=-np.inf 
		elif abs(theta[index])<0.01*abs(means[index]):
			lnpriorprob=-np.inf
	# for index in np.arange(np.shape(theta)[0]):
	# 	if (theta[index]<(means[index]-errors[index])) | (theta[index]>(means[index]+errors[index])):
	# 		lnpriorprob=-np.inf
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

def eigencurves(dict,planetparams,fitfile,plot=False,degree=3,afew=5,burnin=100,nsteps=1000,strict=True,nonegs=True):
	#unpack data dictionary
	# prof = cProfile.Profile()
	# prof.enable()
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
	
	# burnin=300
	# nsteps=3000
	nwalkers=100 #number of walkers	
	if strict:
		biccut=10.
	else:
		biccut=6.
	
	#This file is going to be a 3D numpy array 
	## The shape is (nsamples,n parameters,n waves)
	## where nsamples is the number of posterior MCMC samples
	## n parameters is the number of parameters
	## and n waves is the number of wavelengths looped over
	if isinstance(degree,list):
		# alltheoutput=[]#np.zeros(((nsteps-burnin)*nwalkers,int((degree)**2.),np.shape(waves)[0]))
		bestfitoutput=[]#np.zeros((int((degree)**2.),np.shape(waves)[0]))
	else:
		# alltheoutput=np.zeros(((nsteps-burnin)*nwalkers,int((degree)**2.),np.shape(waves)[0]))
		bestfitoutput=np.zeros((int((degree)**2.),np.shape(waves)[0]))
	
	if np.shape(fluxes)[0]==np.shape(waves)[0]:
		rows=True
	elif np.shape(fluxes)[0]==np.shape(times)[0]:
		rows=False
	else:
		assert (np.shape(fluxes)[0]==np.shape(times)[0]) | (np.shape(fluxes)[0]==np.shape(waves)[0]),"Flux array dimension must match wavelength and time arrays."

	nParamsUsed, ecoeffList, escoreList,elatentList = [], [], [], []
	fullchainarray=[]
	eigencurvecoeffList = []
	for counter in np.arange(np.shape(waves)[0]):
		if isinstance(degree,list):
			# tempoutputholder=np.zeros(((nsteps-burnin)*nwalkers,int((degree[counter])**2.)))
			tempdeg=degree[counter]
		else:
			tempdeg=degree
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

		#alltheoutput[counter,0]=wavelength

		#	Calculate spherical harmonic maps using SPIDERMAN
		lc,t = sh_lcs(t0=t0,per=per,a_abs=a_abs,inc=inc,ecc=ecc,w=planetw,rp=rprs,a=ars,ntimes=eclipsetimes,degree=tempdeg)#degree=lmax+1)#	#model it for the times of the observations
		# for i in range(16):
		# 	plt.figure()
		# 	plt.plot(t,lc[i,:])
		# 	plt.show()

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

		if isinstance(afew,list):#np.shape(afew)[0]==10:
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
				# print(check_negative(degree,params0,ecoeff,wavelength,extent))
				mpfit=leastsq(mpmodel,params0,args=(eclipsetimes,eclipsefluxes,eclipseerrors,elc,np.array(escore),nparams,tempdeg,ecoeff,wavelength,extent,nonegs))
				resid=mpmodel(mpfit[0],eclipsetimes,eclipsefluxes,eclipseerrors,elc,np.array(escore),nparams,tempdeg,ecoeff,wavelength,extent,nonegs)
				throwaway=makeplot(tempdeg,mpfit[0],ecoeff,planetparams,eclipsetimes,eclipsefluxes,eclipseerrors)
				print(check_negative(tempdeg,mpfit[0],ecoeff,wavelength,extent))
				# pdb.set_trace()
				chi2i=np.sum((resid//eclipseerrors)**2.)
				loglike=-0.5*(np.sum((resid//eclipseerrors)**2 + np.log(2.0*np.pi*(eclipseerrors)**2)))
				bici=-2.*loglike + nparams*np.log(np.shape(eclipseerrors)[0])
				tempparams=mpfit[0]
			
				while delbic>biccut:#delbic>10.:
					nparams+=1
					if nparams==15:
						params0=np.zeros(nparams)
						params0[0]=0.0005
						params0[1]=1.
						# print(check_negative(degree,params0,ecoeff,wavelength,extent))
						mpfit=leastsq(mpmodel,params0,args=(eclipsetimes,eclipsefluxes,eclipseerrors,elc,np.array(escore),nparams,tempdeg,ecoeff,wavelength,extent,nonegs))
						chi2f=np.sum((mpmodel(mpfit[0],eclipsetimes,eclipsefluxes,eclipseerrors,elc,np.array(escore),nparams,tempdeg,ecoeff,wavelength,extent,nonegs)//eclipseerrors)**2.)
						delbic=5.
					else:
						# pdb.set_trace()
						params0=np.zeros(nparams)
						params0[0]=0.0005
						params0[1]=1.
						# print(check_negative(degree,params0,ecoeff,wavelength,extent))
						mpfit=leastsq(mpmodel,params0,args=(eclipsetimes,eclipsefluxes,eclipseerrors,elc,np.array(escore),nparams,tempdeg,ecoeff,wavelength,extent,nonegs))
						resid=mpmodel(mpfit[0],eclipsetimes,eclipsefluxes,eclipseerrors,elc,np.array(escore),nparams,tempdeg,ecoeff,wavelength,extent,nonegs)
						throwaway=makeplot(tempdeg,mpfit[0],ecoeff,planetparams,eclipsetimes,eclipsefluxes,eclipseerrors)
						print(check_negative(tempdeg,mpfit[0],ecoeff,wavelength,extent))
						chi2f=np.sum((resid//eclipseerrors)**2.)
						loglike=-0.5*(np.sum((resid//eclipseerrors)**2 + np.log(2.0*np.pi*(eclipseerrors)**2)))
						bicf=-2.*loglike + nparams*np.log(np.shape(eclipseerrors)[0])
						delbic=bici-bicf
						#print(np.sum((resid//eclipseerrors)**2),loglike)
						#print(chi2i,chi2f,bici,bicf)
						print(nparams-2,bici-bicf)#,chi2f-chi2i,bicf-bici)
						print(mpfit[0])
						print(mpfit[0][:-1]-tempparams)
						pdb.set_trace()
						chi2i=chi2f
						bici=bicf
						tempparams=mpfit[0]
				print('BIC criterion says the best number of eigencurves to use is '+str(nparams-3))

			#pdb.set_trace()
				nparams-=1	#need this line back when I change back again
			
			elif ((afew<16)&(afew>=1)):
				nparams=int(afew+2)

			else:	#assert afew is an integer here
				assert afew>1 ,"afew must be an integer 1<=afew<=15!"

		params0=np.zeros(nparams)
		# params0[0]=0.0005
		# params0[1]=1.
		# pdb.set_trace()
		firstfits=np.loadtxt(fitfile+'/besteigenlist_'+str(counter)+'.txt')
		params0=firstfits[(tempdeg-3)*5+(nparams-4),:nparams]
		stepsize=params0*0.001
		# pdb.set_trace()
		# print(check_negative(degree,params0,ecoeff,wavelength,extent))
		# mpfit=leastsq(mpmodel,params0,args=(eclipsetimes,eclipsefluxes[0],eclipseerrors[0],elc,np.array(escore),nparams,tempdeg,ecoeff,wavelength,extent,nonegs))
		# throwaway=makeplot(tempdeg,params0,ecoeff,planetparams,eclipsetimes,eclipsefluxes[0],eclipseerrors[0])
		# print(check_negative(degree,mpfit[0],ecoeff,wavelength,extent))
		#format parameters for mcmc fit
		# pdb.set_trace()
		theta=params0#mpfit[0]
		print(theta)
		ndim=np.shape(theta)[0]	#set number of dimensions
		
		sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(eclipsetimes,eclipsefluxes,eclipseerrors,elc,escore,nparams,tempdeg,ecoeff,wavelength,extent,params0,nonegs))#,pool=mp.Pool(5)
		pos = [theta + stepsize*np.random.randn(ndim) for i in range(nwalkers)]
		print("Running MCMC at {} um".format(waves[counter]))

		bestfit=np.zeros(ndim+1)
		resid=mpmodel(params0,eclipsetimes,eclipsefluxes,eclipseerrors,elc,np.array(escore),nparams,tempdeg,ecoeff,wavelength,extent,nonegs)
		chi2f=np.sum((resid//eclipseerrors)**2.)
		bestfit[1:]=params0
		bestfit[0]=chi2f

		# for i, result in enumerate(sampler.sample(pos, iterations=nsteps,progress=True)):
		# 	if i>burnin:
		# 		for guy in np.arange(nwalkers):
		# 			resid=mpmodel(result.coords[guy],eclipsetimes,eclipsefluxes,eclipseerrors,elc,np.array(escore),nparams,tempdeg,ecoeff,wavelength,extent,nonegs)
		# 			chi2val=np.sum((resid//eclipseerrors)**2.)
		# 			if chi2val<bestfit[0]:
		# 				print('better')
		# 				bestfit[0]=chi2val
		# 				bestfit[1:]=result.coords[guy]
		sampler.run_mcmc(pos,nsteps,progress=True)
		# pdb.set_trace()
		def quantile(x, q):
			return np.percentile(x, [100. * qi for qi in q])
		# pdb.set_trace()
		try:
			print(sampler.get_autocorr_time())
		except:
			print('Could not estimate autocorrelation time.')
		# pdb.set_trace()
		#samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
		#discard burn-in, thin by about half the autocorrelation time (following emcee instructions)
		samples = sampler.get_chain(discard=burnin,thin=int(np.floor(np.mean(sampler.get_autocorr_time()))),flat=True)
		for i in np.arange(ndim):
			bestfit[i+1]=quantile(samples[:,i],[0.5])[0]
		# pdb.set_trace()
		resid=mpmodel(bestfit[1:],eclipsetimes,eclipsefluxes,eclipseerrors,elc,np.array(escore),nparams,tempdeg,ecoeff,wavelength,extent,nonegs)
		chi2f=np.sum((resid//eclipseerrors)**2.)
		# pdb.set_trace()
		# templc=makeplot(tempdeg,bestfit[1:],ecoeff,planetparams,eclipsetimes,eclipsefluxes[0],eclipseerrors[0])
		bestfit[0]=chi2f
		loglike=-0.5*(np.sum((resid//eclipseerrors)**2 + np.log(2.0*np.pi*(eclipseerrors)**2)))
		bicf=-2.*loglike + nparams*np.log(np.shape(eclipseerrors)[0])

		
		fullchain=sampler.chain
		fullchainarray.append(fullchain)

		bestcoeffs=bestfit[1:]

		#make an eclipse lightcurve for one wavelength
		plotyvals=bestcoeffs[0]*elc[0,:]+bestcoeffs[1]
		for i in np.arange(int(np.shape(bestcoeffs)[0]-2)):
			plotyvals+=bestcoeffs[i+2]*escore[i,:]
		
		# translate coefficients
		fcoeffbest=np.zeros_like(ecoeff)
		for i in np.arange(np.shape(bestcoeffs)[0]-2):
			fcoeffbest[:,i] = bestcoeffs[i+2]*ecoeff[:,i]

		# how to go from coefficients to best fit map
		spheresbest=np.zeros(int((tempdeg)**2.))
		for j in range(0,len(fcoeffbest)):
			for i in range(1,int((tempdeg)**2.)):
				spheresbest[i] += fcoeffbest.T[j,2*i-1]-fcoeffbest.T[j,2*(i-1)]
		spheresbest[0] = bestcoeffs[0]#c0_best
		if isinstance(degree,list):
			bestfitoutput.append(spheresbest)
		else:
			bestfitoutput[:,counter]=spheresbest

		# for sampnum in np.arange(np.shape(samples)[0]): #THIS IS THE PART THAT TAKES FOREVER
		# 	fcoeff=np.zeros_like(ecoeff)
		# 	for i in np.arange(np.shape(samples)[1]-2):
		# 		fcoeff[:,i] = samples[sampnum,i+2]*ecoeff[:,i]

		# 	# how to go from coefficients to best fit map
		# 	spheres=np.zeros(int((tempdeg)**2.))
		# 	for j in range(0,len(fcoeff)):
		# 		for i in range(1,int((tempdeg)**2.)):
		# 			spheres[i] += fcoeff.T[j,2*i-1]-fcoeff.T[j,2*(i-1)]
		# 	spheres[0] = samples[sampnum,0]#bestcoeffs[0]#c0_best	
			
		# 	if isinstance(degree,list):
		# 		tempoutputholder[sampnum,:]=spheres
		# 	else:
		# 		alltheoutput[sampnum,:,counter]=spheres

		if plot:
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

			params0.degree=tempdeg	#maximum harmonic degree
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

		nParamsUsed.append(nparams)
		ecoeffList.append(ecoeff)
		escoreList.append(escore)
		eigencurvecoeffList.append(samples)
		elatentList.append(elatent)
		# if isinstance(degree,list):
		# 	alltheoutput.append(tempoutputholder)

	# prof.disable()
	# prof.print_stats()	
	
	finaldict={'wavelength (um)':waves,\
		'best fit coefficients':bestfitoutput,'N Params Used':nParamsUsed,\
				'ecoeffList': ecoeffList,'escoreList': escoreList,'elc': elc,\
				'eigencurve coefficients':eigencurvecoeffList,'BIC':bicf,\
				'elatentList':elatentList,'fullchainarray':fullchainarray,\
				'degree':degree,'best spheres':spheresbest} #'spherical coefficients':alltheoutput,
	return finaldict
