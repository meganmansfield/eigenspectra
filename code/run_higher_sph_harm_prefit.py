"""

"""
# coding: utf-8

# In[7]:


# Import functions
import numpy as np
import eigencurves_prefit
import os

from importlib import import_module


def run_lc_fit(datadict,planetparams,norder=3,lcName='lcname',saveDir='saveDir',afew=10,burnin=100,nsteps=1000,plot=False,strict=True,nonegs=True):
    """
    Run eigencurves fitting to retrieve a planet map
    
    Parameters:
    -----------
    norder: int
        Number of the spherical harmonic to fit
    usePath: str
        Path to the npz file for light curves
    afew: int
        How many eigencurves to fit for (if >=10, the best fitting algorithm will run and select the number of eigencurves for you)
    
    Outputs:
    -----------
    None: all data are saved to data/sph_harmonic_coefficients_full_samples
    """
    
    if os.path.exists(saveDir) == False:
        os.makedirs(saveDir)
    if isinstance(afew,int):
        if afew>=16:
            outputNPZ = '{}/spherearray_deg_{}_eigenfit.npz'.format(saveDir,norder)
        else:
            outputNPZ = '{}/spherearray_deg_{}_eigen_{}.npz'.format(saveDir,norder,afew)
    else:
        outputNPZ = '{}/spherearray_deg_{}.npz'.format(saveDir,norder)
    
    if os.path.exists(outputNPZ):
        print("Found the previously-run file {}. Now exiting".format(outputNPZ))
        return
    else:
        print("No previous run found, so running MCMC.")
        print("This can take a long time, especially for higher spherical harmonic orders")

        # ### Fit eigencurves to lightcurve
        print("Fitting eigencurves now for order {}".format(norder))
        spherearray = eigencurves_prefit.eigencurves(datadict,planetparams,plot=plot,degree=norder,afew=afew,burnin=burnin,nsteps=nsteps,strict=strict,nonegs=nonegs)#_starry
        # spherearray is an array of wavelength x SH coefficents
    
        np.savez(outputNPZ,spherearray)
