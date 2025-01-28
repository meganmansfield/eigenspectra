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

def eigencurves(dict,planetparams,plot=False,degree=3,afew=5,burnin=100,nsteps=1000,strict=True,nonegs=True):
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
	
	# alltheoutput=np.zeros(((nsteps-burnin)*nwalkers,int((degree)**2.),np.shape(waves)[0]))
	# bestfitoutput=np.zeros((int((degree)**2.),np.shape(waves)[0]))
	
	if np.shape(fluxes)[0]==np.shape(waves)[0]:
		rows=True
	elif np.shape(fluxes)[0]==np.shape(times)[0]:
		rows=False
	else:
		assert (np.shape(fluxes)[0]==np.shape(times)[0]) | (np.shape(fluxes)[0]==np.shape(waves)[0]),"Flux array dimension must match wavelength and time arrays."

	nParamsUsed = np.zeros(np.shape(waves)[0])
	# fullchainarray=[]
	# eigencurvecoeffList = []
	eigencurvecoeffList=np.zeros((np.shape(waves)[0],(nsteps-burnin)*nwalkers,int(afew+2))) #FINDME: FIX
	biclist=np.zeros(np.shape(waves)[0])
	for counter in np.arange(np.shape(waves)[0]):
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
		lc,t = sh_lcs(t0=t0,per=per,a_abs=a_abs,inc=inc,ecc=ecc,w=planetw,rp=rprs,a=ars,ntimes=eclipsetimes,degree=degree)	#model it for the times of the observations

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

		# if isinstance(afew,list):#np.shape(afew)[0]==10:
		# 	print('Gave an array of afew values')
		# 	if not np.shape(waves)[0]==np.shape(afew)[0]:
		# 		assert (np.shape(waves)[0]==np.shape(afew)[0]), "Array of afew values must be the same length as the number of wavelength bins, which is"+str(np.shape(waves)[0])
		# 	nparams=int(afew[counter]+2)
		# else:
		# 	if not isinstance(afew,int):
		# 		assert isinstance(afew,int), "afew must be an integer >=1!"
		# 	elif afew>=16:
		# 		print('Performing fit for best number of eigencurves to use.')
		# 		delbic=20.
		# 		nparams=4
		# 		params0=np.zeros(nparams)
		# 		params0[0]=0.0005
		# 		params0[1]=1.
		# 		params0[3]=0.0001
		# 		# print(check_negative(degree,params0,ecoeff,wavelength,extent))
		# 		# throwaway=makeplot(degree,params0,ecoeff,planetparams,eclipsetimes,eclipsefluxes,eclipseerrors)
		# 		# mpfit=leastsq(mpmodel,params0,args=(eclipsetimes,eclipsefluxes,eclipseerrors,elc,np.array(escore),nparams,degree,ecoeff,wavelength,extent,nonegs))
				
		# 		results = minimize(neg_lnprob, params0, args=(eclipsetimes,eclipsefluxes,\
		# 			eclipseerrors,elc,np.array(escore),nparams,degree,ecoeff,wavelength,\
		# 			extent,nonegs),method='Nelder-Mead',tol=1e-6,\
		# 		options={'maxiter':None})
		# 		fit_params = results.x
		# 		# resid=mpmodel(mpfit[0],eclipsetimes,eclipsefluxes,eclipseerrors,elc,np.array(escore),nparams,degree,ecoeff,wavelength,extent,nonegs)
		# 		# throwaway=makeplot(degree,mpfit[0],ecoeff,planetparams,eclipsetimes,eclipsefluxes,eclipseerrors)
		# 		# print(check_negative(degree,mpfit[0],ecoeff,wavelength,extent))
		# 		resid=mpmodel(fit_params,eclipsetimes,eclipsefluxes,eclipseerrors,elc,np.array(escore),nparams,degree,ecoeff,wavelength,extent,nonegs)
		# 		# throwaway=makeplot(degree,fit_params,ecoeff,planetparams,eclipsetimes,eclipsefluxes,eclipseerrors)
		# 		print(check_negative(degree,fit_params,ecoeff,wavelength,extent))
		# 		fit_params_old = params0
		# 		del_params = abs((fit_params_old-fit_params)/(fit_params))
		# 		print(np.max(del_params))
		# 		chi2i=np.sum((resid//eclipseerrors)**2.)
		# 		loglike=-0.5*(np.sum((resid//eclipseerrors)**2 + np.log(2.0*np.pi*(eclipseerrors)**2)))
		# 		bicfit=-2.*loglike + nparams*np.log(np.shape(eclipseerrors)[0])
		# 		print(bicfit)
		# 		pdb.set_trace()
		# 		while any(del_params>0.01):
		# 			fit_params_old = fit_params
		# 			params0 = fit_params
		# 			results = minimize(neg_lnprob, params0, args=(eclipsetimes,eclipsefluxes,\
		# 				eclipseerrors,elc,np.array(escore),nparams,degree,ecoeff,wavelength,\
		# 				extent,nonegs),method='Nelder-Mead',tol=1e-6,\
		# 			options={'maxiter':None})
		# 			fit_params = results.x
		# 			resid=mpmodel(fit_params,eclipsetimes,eclipsefluxes,eclipseerrors,elc,np.array(escore),nparams,degree,ecoeff,wavelength,extent,nonegs)
		# 			# throwaway=makeplot(degree,fit_params,ecoeff,planetparams,eclipsetimes,eclipsefluxes,eclipseerrors)
		# 			print(check_negative(degree,fit_params,ecoeff,wavelength,extent))
		# 			del_params = abs((fit_params_old-fit_params)/(fit_params))
		# 			print(np.max(del_params))
		# 			chi2i=np.sum((resid//eclipseerrors)**2.)
		# 			loglike=-0.5*(np.sum((resid//eclipseerrors)**2 + np.log(2.0*np.pi*(eclipseerrors)**2)))
		# 			bicfit=-2.*loglike + nparams*np.log(np.shape(eclipseerrors)[0])
		# 			print(bicfit)
		# 			pdb.set_trace()
		# 		pdb.set_trace()
		# 		tempparams=fit_params#mpfit[0]
		# 		bici=bicfit
		# 		#######FINDME: COMMENTING OUT EMCEE############
		# 		# theta = fit_params
		# 		# ndim=np.shape(theta)[0]	#set number of dimensions
		# 		# stepsize=0.001*theta
		# 		# sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(eclipsetimes,eclipsefluxes,eclipseerrors,elc,escore,nparams,degree,ecoeff,wavelength,extent,nonegs),pool=mp.Pool(5))
		# 		# pos = [theta + stepsize*np.random.randn(ndim) for i in range(nwalkers)]
		# 		# # print("Running MCMC at {} um".format(waves[counter]))
		# 		# sampler.run_mcmc(pos,nsteps,progress=True)
		# 		# try:
		# 		# 	print(sampler.get_autocorr_time())
		# 		# except:
		# 		# 	print('Could not estimate autocorrelation time.')
		# 		# samples = sampler.get_chain(discard=burnin,thin=int(np.floor(np.mean(sampler.get_autocorr_time()))),flat=True)
		# 		# bestfit=np.zeros(len(theta))
		# 		# for i in np.arange(ndim):
		# 		# 	bestfit[i]=quantile(samples[:,i],[0.5])[0]
		# 		# resid=mpmodel(bestfit,eclipsetimes,eclipsefluxes,eclipseerrors,elc,np.array(escore),nparams,degree,ecoeff,wavelength,extent,nonegs)
		# 		# chi2i=np.sum((resid//eclipseerrors)**2.)
		# 		# loglike=-0.5*(np.sum((resid//eclipseerrors)**2 + np.log(2.0*np.pi*(eclipseerrors)**2)))
		# 		# bici=-2.*loglike + nparams*np.log(np.shape(eclipseerrors)[0])
		# 		# print(bestfit-fit_params)
		# 		# print(bicfit,bici,bicfit-bici)
		# 		# tempparams=bestfit
		# 		# pdb.set_trace()

		# 		while delbic>biccut:#delbic>10.:
		# 			nparams+=1
		# 			if nparams==15:
		# 				params0=np.zeros(nparams)
		# 				params0[0]=0.0005
		# 				params0[1]=1.
		# 				params0[3]=0.0001
		# 				# print(check_negative(degree,params0,ecoeff,wavelength,extent))
		# 				# mpfit=leastsq(mpmodel,params0,args=(eclipsetimes,eclipsefluxes,eclipseerrors,elc,np.array(escore),nparams,degree,ecoeff,wavelength,extent,nonegs))
		# 				# chi2f=np.sum((mpmodel(mpfit[0],eclipsetimes,eclipsefluxes,eclipseerrors,elc,np.array(escore),nparams,degree,ecoeff,wavelength,extent,nonegs)//eclipseerrors)**2.)
						
		# 				results = minimize(neg_lnprob, params0, args=(eclipsetimes,eclipsefluxes,\
		# 					eclipseerrors,elc,np.array(escore),nparams,degree,ecoeff,wavelength,\
		# 					extent,nonegs),method='Nelder-Mead',tol=1e-6,\
		# 				options={'maxiter':None})
		# 				fit_params = results.x
		# 				resid=mpmodel(fit_params,eclipsetimes,eclipsefluxes,eclipseerrors,elc,np.array(escore),nparams,degree,ecoeff,wavelength,extent,nonegs)
		# 				# throwaway=makeplot(degree,fit_params,ecoeff,planetparams,eclipsetimes,eclipsefluxes,eclipseerrors)
		# 				print(check_negative(degree,fit_params,ecoeff,wavelength,extent))
		# 				fit_params_old = params0
		# 				del_params = abs((fit_params_old-fit_params)/(fit_params))
		# 				print(np.max(del_params))
		# 				while any(del_params>0.01):
		# 					fit_params_old = fit_params
		# 					params0 = fit_params
		# 					results = minimize(neg_lnprob, params0, args=(eclipsetimes,eclipsefluxes,\
		# 						eclipseerrors,elc,np.array(escore),nparams,degree,ecoeff,wavelength,\
		# 						extent,nonegs),method='Nelder-Mead',tol=1e-6,\
		# 					options={'maxiter':None})
		# 					fit_params = results.x
		# 					resid=mpmodel(fit_params,eclipsetimes,eclipsefluxes,eclipseerrors,elc,np.array(escore),nparams,degree,ecoeff,wavelength,extent,nonegs)
		# 					# throwaway=makeplot(degree,fit_params,ecoeff,planetparams,eclipsetimes,eclipsefluxes,eclipseerrors)
		# 					print(check_negative(degree,fit_params,ecoeff,wavelength,extent))
		# 					del_params = abs((fit_params_old-fit_params)/(fit_params))
		# 					print(np.max(del_params))
		# 				chi2f=np.sum((resid//eclipseerrors)**2.)
		# 				delbic=5.
		# 			else:
		# 				# pdb.set_trace()
		# 				params0=np.zeros(nparams)
		# 				params0[0]=0.0005
		# 				params0[1]=1.
		# 				params0[3]=0.0001
		# 				# print(check_negative(degree,params0,ecoeff,wavelength,extent))
		# 				# mpfit=leastsq(mpmodel,params0,args=(eclipsetimes,eclipsefluxes,eclipseerrors,elc,np.array(escore),nparams,degree,ecoeff,wavelength,extent,nonegs))
		# 				# resid=mpmodel(mpfit[0],eclipsetimes,eclipsefluxes,eclipseerrors,elc,np.array(escore),nparams,degree,ecoeff,wavelength,extent,nonegs)
		# 				# throwaway=makeplot(degree,mpfit[0],ecoeff,planetparams,eclipsetimes,eclipsefluxes,eclipseerrors)
		# 				# print(check_negative(degree,mpfit[0],ecoeff,wavelength,extent))
		# 				results = minimize(neg_lnprob, params0, args=(eclipsetimes,eclipsefluxes,\
		# 					eclipseerrors,elc,np.array(escore),nparams,degree,ecoeff,wavelength,\
		# 					extent,nonegs),method='Nelder-Mead',tol=1e-6,\
		# 				options={'maxiter':None})
		# 				fit_params = results.x
		# 				resid=mpmodel(fit_params,eclipsetimes,eclipsefluxes,eclipseerrors,elc,np.array(escore),nparams,degree,ecoeff,wavelength,extent,nonegs)
		# 				# throwaway=makeplot(degree,fit_params,ecoeff,planetparams,eclipsetimes,eclipsefluxes,eclipseerrors)
		# 				print(check_negative(degree,fit_params,ecoeff,wavelength,extent))
		# 				fit_params_old = params0
		# 				del_params = abs((fit_params_old-fit_params)/(fit_params))
		# 				print(np.max(del_params))
		# 				chi2i=np.sum((resid//eclipseerrors)**2.)
		# 				loglike=-0.5*(np.sum((resid//eclipseerrors)**2 + np.log(2.0*np.pi*(eclipseerrors)**2)))
		# 				bicfit=-2.*loglike + nparams*np.log(np.shape(eclipseerrors)[0])
		# 				print(bicfit)
		# 				pdb.set_trace()
		# 				while any(del_params>0.01):
		# 					fit_params_old = fit_params
		# 					params0 = fit_params
		# 					results = minimize(neg_lnprob, params0, args=(eclipsetimes,eclipsefluxes,\
		# 						eclipseerrors,elc,np.array(escore),nparams,degree,ecoeff,wavelength,\
		# 						extent,nonegs),method='Nelder-Mead',tol=1e-6,\
		# 					options={'maxiter':None})
		# 					fit_params = results.x
		# 					resid=mpmodel(fit_params,eclipsetimes,eclipsefluxes,eclipseerrors,elc,np.array(escore),nparams,degree,ecoeff,wavelength,extent,nonegs)
		# 					# throwaway=makeplot(degree,fit_params,ecoeff,planetparams,eclipsetimes,eclipsefluxes,eclipseerrors)
		# 					print(check_negative(degree,fit_params,ecoeff,wavelength,extent))
		# 					del_params = abs((fit_params_old-fit_params)/(fit_params))
		# 					print(np.max(del_params))
		# 					chi2i=np.sum((resid//eclipseerrors)**2.)
		# 					loglike=-0.5*(np.sum((resid//eclipseerrors)**2 + np.log(2.0*np.pi*(eclipseerrors)**2)))
		# 					bicfit=-2.*loglike + nparams*np.log(np.shape(eclipseerrors)[0])
		# 					print(bicfit)
		# 					pdb.set_trace()
		# 				# chi2i=np.sum((resid//eclipseerrors)**2.)
		# 				# loglike=-0.5*(np.sum((resid//eclipseerrors)**2 + np.log(2.0*np.pi*(eclipseerrors)**2)))
		# 				# bicfit=-2.*loglike + nparams*np.log(np.shape(eclipseerrors)[0])
						
		# 				# pdb.set_trace()

		# 				#########FINDME: COMMENT OUT EMCEE
		# 				# theta = fit_params
		# 				# ndim=np.shape(theta)[0]	#set number of dimensions
		# 				# stepsize=0.001*theta
		# 				# sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(eclipsetimes,eclipsefluxes,eclipseerrors,elc,escore,nparams,degree,ecoeff,wavelength,extent,nonegs),pool=mp.Pool(5))
		# 				# pos = [theta + stepsize*np.random.randn(ndim) for i in range(nwalkers)]
		# 				# # print("Running MCMC at {} um".format(waves[counter]))
		# 				# sampler.run_mcmc(pos,nsteps,progress=True)
		# 				# try:
		# 				# 	print(sampler.get_autocorr_time())
		# 				# except:
		# 				# 	print('Could not estimate autocorrelation time.')
		# 				# samples = sampler.get_chain(discard=burnin,thin=int(np.floor(np.mean(sampler.get_autocorr_time()))),flat=True)
		# 				# bestfit=np.zeros(len(theta))
		# 				# for i in np.arange(ndim):
		# 				# 	bestfit[i]=quantile(samples[:,i],[0.5])[0]
		# 				# resid=mpmodel(bestfit,eclipsetimes,eclipsefluxes,eclipseerrors,elc,np.array(escore),nparams,degree,ecoeff,wavelength,extent,nonegs)
		# 				# # chi2i=np.sum((resid//eclipseerrors)**2.)
		# 				# # loglike=-0.5*(np.sum((resid//eclipseerrors)**2 + np.log(2.0*np.pi*(eclipseerrors)**2)))
		# 				# # bici=-2.*loglike + nparams*np.log(np.shape(eclipseerrors)[0])
		# 				# print(bestfit-fit_params)
		# 				# # tempparams=bestfit

		# 				# chi2f=np.sum((resid//eclipseerrors)**2.)
		# 				# loglike=-0.5*(np.sum((resid//eclipseerrors)**2 + np.log(2.0*np.pi*(eclipseerrors)**2)))
		# 				# bicf=-2.*loglike + nparams*np.log(np.shape(eclipseerrors)[0])
		# 				# delbic=bici-bicf
		# 				delbic=bici-bicfit
		# 				print(bicfit)#,bicf,bicfit-bicf)
		# 				#print(np.sum((resid//eclipseerrors)**2),loglike)
		# 				#print(chi2i,chi2f,bici,bicf)
		# 				print(nparams-2,bici-bicf,bici,bicf)#,chi2f-chi2i,bicf-bici)
		# 				print(bestfit)#(mpfit[0])
		# 				print(bestfit[:-1]-tempparams)#(mpfit[0][:-1]-tempparams)
		# 				# pdb.set_trace()
		# 				chi2i=chi2f
		# 				bici=bicf
		# 				tempparams=bestfit#fit_params#mpfit[0]
		# 				pdb.set_trace()
		# 		print('BIC criterion says the best number of eigencurves to use is '+str(nparams-3))

		# 		# pdb.set_trace()
		# 		nparams-=1	#need this line back when I change back again
			
		# 	elif ((afew<16)&(afew>=1)):
		# 		nparams=int(afew+2)

		# 	else:	#assert afew is an integer here
		# 		assert afew>1 ,"afew must be an integer 1<=afew<=15!"

		nparams=int(afew+2)
		params0=np.zeros(nparams)
		params0[0]=0.0005
		params0[1]=1.
		params0[3]=0.0001
		# params0[0]=0.0005
		# params0[1]=1.
		# if nparams==4:
		# 	params0=np.array([1.6266e-4,1.000099,-7.9237e-7,8.7091e-5])
		# 	stepsize=np.array([1e-7,1e-3,1e-10,1e-8])
		# elif nparams==5:
		# 	params0=np.array([1.6370e-4,1.000103,-6.1765e-7,8.8249e-5,2.3366e-5])
		# 	stepsize=np.array([1e-7,1e-3,1e-10,1e-8,1e-8])
		
		# print(check_negative(degree,params0,ecoeff,wavelength,extent))
		# mpfit=leastsq(mpmodel,params0,args=(eclipsetimes,eclipsefluxes,eclipseerrors,elc,np.array(escore),nparams,degree,ecoeff,wavelength,extent,nonegs))
		# throwaway=makeplot(degree,mpfit[0],ecoeff,planetparams,eclipsetimes,eclipsefluxes,eclipseerrors)
		# print(check_negative(degree,mpfit[0],ecoeff,wavelength,extent))
		#format parameters for mcmc fit

		results = minimize(neg_lnprob, params0, args=(eclipsetimes,eclipsefluxes,\
			eclipseerrors,elc,np.array(escore),nparams,degree,ecoeff,wavelength,\
			extent,nonegs),method='Nelder-Mead',tol=1e-6,\
		options={'maxiter':None})
		fit_params = results.x
		resid=mpmodel(fit_params,eclipsetimes,eclipsefluxes,eclipseerrors,elc,np.array(escore),nparams,degree,ecoeff,wavelength,extent,nonegs)
		print(check_negative(degree,fit_params,ecoeff,wavelength,extent))
		fit_params_old = params0
		del_params = abs((fit_params_old-fit_params)/(fit_params))
		print(np.max(del_params))
		chi2i=np.sum((resid//eclipseerrors)**2.)
		loglike=-0.5*(np.sum((resid//eclipseerrors)**2 + np.log(2.0*np.pi*(eclipseerrors)**2)))
		bici=-2.*loglike + nparams*np.log(np.shape(eclipseerrors)[0])
		print(bici)
		bicf=bici
		# pdb.set_trace()
		while any(del_params>0.01):
			bici=bicf
			fit_params_old = fit_params
			params0 = fit_params
			results = minimize(neg_lnprob, params0, args=(eclipsetimes,eclipsefluxes,\
				eclipseerrors,elc,np.array(escore),nparams,degree,ecoeff,wavelength,\
				extent,nonegs),method='Nelder-Mead',tol=1e-6,\
			options={'maxiter':None})
			fit_params = results.x
			resid=mpmodel(fit_params,eclipsetimes,eclipsefluxes,eclipseerrors,elc,np.array(escore),nparams,degree,ecoeff,wavelength,extent,nonegs)
			# throwaway=makeplot(degree,fit_params,ecoeff,planetparams,eclipsetimes,eclipsefluxes,eclipseerrors)
			print(check_negative(degree,fit_params,ecoeff,wavelength,extent))
			del_params = abs((fit_params_old-fit_params)/(fit_params))
			print(np.max(del_params))
			chi2i=np.sum((resid//eclipseerrors)**2.)
			loglike=-0.5*(np.sum((resid//eclipseerrors)**2 + np.log(2.0*np.pi*(eclipseerrors)**2)))
			bicf=-2.*loglike + nparams*np.log(np.shape(eclipseerrors)[0])
			print(bicf)
			# pdb.set_trace()
		# pdb.set_trace()
		# chi2i=np.sum((resid//eclipseerrors)**2.)
		# loglike=-0.5*(np.sum((resid//eclipseerrors)**2 + np.log(2.0*np.pi*(eclipseerrors)**2)))
		bicf=-2.*loglike + nparams*np.log(np.shape(eclipseerrors)[0])
		bestcoeffs=fit_params#mpfit[0]
		# bici=bicf
		# ndim=np.shape(theta)[0]	#set number of dimensions
		# stepsize=0.001*theta
		# sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(eclipsetimes,eclipsefluxes,eclipseerrors,elc,escore,nparams,degree,ecoeff,wavelength,extent,nonegs),pool=mp.Pool(5))
		# pos = [theta + stepsize*np.random.randn(ndim) for i in range(nwalkers)]
		# # print("Running MCMC at {} um".format(waves[counter]))
		# sampler.run_mcmc(pos,nsteps,progress=True)
		# try:
		# 	print(sampler.get_autocorr_time())
		# except:
		# 	print('Could not estimate autocorrelation time.')
		
		# theta=fit_params#mpfit[0]
		# print(theta)
		# ndim=np.shape(theta)[0]	#set number of dimensions
		# stepsize=0.001*theta #0.001
		# sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(eclipsetimes,eclipsefluxes,eclipseerrors,elc,escore,nparams,degree,ecoeff,wavelength,extent,nonegs),pool=mp.Pool(5))
		# pos = [theta + stepsize*np.random.randn(ndim) for i in range(nwalkers)]
		# print("Running MCMC at {} um".format(waves[counter]))

		# bestfit=np.zeros(ndim+1)
		# bestfit[0]=10.**8

		# burnin=0
		# nsteps=300
		# for i, result in enumerate(sampler.sample(pos, iterations=nsteps,progress=True)):
		# 	if i>burnin:
		# 		for guy in np.arange(nwalkers):
		# 			resid=mpmodel(result.coords[guy],eclipsetimes,eclipsefluxes,eclipseerrors,elc,np.array(escore),nparams,degree,ecoeff,wavelength,extent,nonegs)
		# 			chi2val=np.sum((resid//eclipseerrors)**2.)
		# 			if chi2val<bestfit[0]:
		# 				bestfit[0]=chi2val
		# 				bestfit[1:]=result.coords[guy]
		# 				loglike=-0.5*(np.sum((resid//eclipseerrors)**2 + np.log(2.0*np.pi*(eclipseerrors)**2)))
		# 				bicf=-2.*loglike + nparams*np.log(np.shape(eclipseerrors)[0])
		# sampler.run_mcmc(pos,nsteps,progress=True)


		# try:
		# 	print(sampler.get_autocorr_time())
		# except:
		# 	print('Could not estimate autocorrelation time.')
		# # pdb.set_trace()
		# samples = sampler.get_chain(discard=burnin,flat=True)#thin=int(np.floor(np.mean(sampler.get_autocorr_time()))),
		# tbestfit=np.zeros(ndim+1)
		# for i in np.arange(ndim):
		# 	tbestfit[i+1]=quantile(samples[:,i],[0.5])[0]

		# tresid=mpmodel(tbestfit[1:],eclipsetimes,eclipsefluxes,eclipseerrors,elc,np.array(escore),nparams,degree,ecoeff,wavelength,extent,nonegs)
		# tchi2f=np.sum((tresid//eclipseerrors)**2.)
		# tbestfit[0]=tchi2f
		# tloglike=-0.5*(np.sum((tresid//eclipseerrors)**2 + np.log(2.0*np.pi*(eclipseerrors)**2)))
		# tbicf=-2.*tloglike + nparams*np.log(np.shape(eclipseerrors)[0])
		print(bici,bicf,bici-bicf)
		# print(bici,tbicf,bici-tbicf)
		# pdb.set_trace()
		# fullchain=sampler.chain
		# # fullchainarray.append(fullchain)

		# bestcoeffs=bestfit[1:]
		# bestcoeffs=fit_params

		#make an eclipse lightcurve for one wavelength
		plotyvals=bestcoeffs[0]*elc[0,:]+bestcoeffs[1]
		for i in np.arange(int(np.shape(bestcoeffs)[0]-2)):
			plotyvals+=bestcoeffs[i+2]*escore[i,:]

		# translate coefficients
		fcoeffbest=np.zeros_like(ecoeff)
		for i in np.arange(np.shape(bestcoeffs)[0]-2):
			fcoeffbest[:,i] = bestcoeffs[i+2]*ecoeff[:,i]

		# how to go from coefficients to best fit map
		spheresbest=np.zeros(int((degree)**2.))
		for j in range(0,len(fcoeffbest)):
			for i in range(1,int((degree)**2.)):
				spheresbest[i] += fcoeffbest.T[j,2*i-1]-fcoeffbest.T[j,2*(i-1)]
		spheresbest[0] = bestcoeffs[0]#c0_best
		# bestfitoutput[:,counter]=spheresbest

		# for sampnum in np.arange(np.shape(samples)[0]):
		# 	fcoeff=np.zeros_like(ecoeff)
		# 	for i in np.arange(np.shape(samples)[1]-2):
		# 		fcoeff[:,i] = samples[sampnum,i+2]*ecoeff[:,i]

		# 	# how to go from coefficients to best fit map
		# 	spheres=np.zeros(int((degree)**2.))
		# 	for j in range(0,len(fcoeff)):
		# 		for i in range(1,int((degree)**2.)):
		# 			spheres[i] += fcoeff.T[j,2*i-1]-fcoeff.T[j,2*(i-1)]
		# 	spheres[0] = samples[sampnum,0]#bestcoeffs[0]#c0_best	
			
			# alltheoutput[sampnum,:,counter]=spheres

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

			params0.degree=degree	#maximum harmonic degree
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

		nParamsUsed[counter]=nparams
		# eigencurvecoeffList[counter,:,:]=samples
		biclist[counter]=bicf

		
	
	# finaldict={'wavelength (um)':waves,'spherical coefficients':alltheoutput,'best fit coefficients':bestfitoutput,'N Params Used':nParamsUsed,
	# 			'ecoeffList': ecoeffList,'escoreList': escoreList,'elc': elc,'eigencurve coefficients':eigencurvecoeffList,'BIC':bicf,'elatentList':elatentList,'fullchainarray':fullchainarray}
	finaldict={'wavelength (um)':waves,'N Params Used':nParamsUsed,'eigencurve coefficients':eigencurvecoeffList,'BIC':biclist}
	return finaldict
