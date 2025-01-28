import numpy as np
import pdb


def bin_eigenspectra(maps, kgroup_draws, lats, lons, ngroups):
    '''
    Converts a grid of spectra into eigenspectra defined by kgroups.

    Parameters
    ----------
    spectra : array of Fp/Fs (axes: wavelengths x lat x lon)
    kgroups : array of group indices (ints) from 0 to k-1 (axes: lat x lon)

    Returns
    -------
    eigenspectra : list of k spectra, averaged over each group
    '''

    # Calculate the number of groups
    # assuming kgroups contains integers from 0 to ngroups
    # pdb.set_trace()
    # ngroups = np.max(kgroups)+1

    # nbins = spectra.shape[0]  # number of wavelength bins
    # # spectra = spectra.reshape(nbins, -1)  # Flatten over (lat x lon)
    # # kgroups = kgroups.reshape(-1)
    # # pdb.set_trace()
    # eigenspectra = []
    # eigenlist = [[]]*ngroups
    # centlon=0.
    # centlat=0.
    # dellat=np.diff(lats[:,0])[0]
    # dellon=np.diff(lons[0,:])[0]
    # centlat_sin = np.sin(centlat)
    # centlat_cos = np.cos(centlat)
    # vis = centlat_sin * np.sin(lats) + centlat_cos * np.cos(lats) * \
    #         np.cos(lons - centlon)
    # vis[vis <= 0.] = 0.
    # for g in range(ngroups):
    #     # ingroup = (kgroups == g).astype(int)
    #     # eigenspec = np.sum(spectra*ingroup, axis=1)/np.sum(ingroup)
    #     # # eigenspec is the mean of spectra in group
    #     # eigenspectra.append(eigenspec)
    #     # ingroups2 = (kgroups == g)
    #     # eigenlist[g]=spectra[:,ingroups2]

    #     #visibility weighting assuming center-of-eclipse (longitude=0,latitude=0 at central point)
        
    #     ingroup = np.where(kgroups==g)
    #     # pdb.set_trace()
    #     eigenspec = np.sum(spectra[:,ingroup[0],ingroup[1]] * vis[ingroup[0],ingroup[1]] \
    #         * np.cos(lats[ingroup[0],ingroup[1]]),axis=1) * dellat * dellon
    #     eigenspectra.append(eigenspec)
    #     eigenlist[g]=spectra[:,ingroup[0],ingroup[1]]
    # # pdb.set_trace()
    kgroups = np.mean(kgroup_draws, axis=0)
    perpointspec = np.mean(maps,axis=0)
    perpointerr = np.std(maps,axis=0)
    nearestint= np.round(kgroups,decimals=0)
    nwaves=np.shape(maps)[1]
    eigenspec = np.zeros((ngroups,nwaves))
    eigenerr = np.zeros((ngroups,nwaves))
    for g in range(ngroups):
        ingroup = np.where(nearestint==g)
        if len(ingroup[0])==0:
            eigenspec[g,:] = np.zeros(nwaves)
            eigenerr[g,:] = np.zeros(nwaves)
        else:
            eigenspec[g,:] = np.sum(perpointspec[:,ingroup[0],ingroup[1]] * \
                np.cos(lats[ingroup[0],ingroup[1]]),axis=1)/np.sum(np.cos(lats[ingroup[0],ingroup[1]]))
            eigenerr[g,:] = np.sum(perpointerr[:,ingroup[0],ingroup[1]] * \
                np.cos(lats[ingroup[0],ingroup[1]]),axis=1)/np.sum(np.cos(lats[ingroup[0],ingroup[1]]))
    

    #integrate over full sphere
    centlon=0.
    centlat=0.
    dellat=np.diff(lats[:,0])[0]
    dellon=np.diff(lons[0,:])[0]
    centlat_sin = np.sin(centlat)
    centlat_cos = np.cos(centlat)
    vis = centlat_sin * np.sin(lats) + centlat_cos * np.cos(lats) * \
            np.cos(lons - centlon)
    vis[vis <= 0.] = 0.
    integratedspec = np.zeros((ngroups,nwaves))
    integratederr = np.zeros((ngroups,nwaves))
    for g in range(ngroups):
        integratedspec[g,:] = eigenspec[g,:]*np.sum(vis*np.cos(lats)) * dellat * dellon
        integratederr[g,:] = eigenerr[g,:]*np.sum(vis*np.cos(lats)) * dellat * dellon
    fullplanetspec=np.zeros(nwaves)
    fullplaneterr=np.zeros(nwaves)
    for i in np.arange(nwaves):
        fullplanetspec[i] = np.sum(perpointspec[i,:,:] * vis * np.cos(lats)) * dellat * dellon
        fullplaneterr[i] = np.sum(perpointerr[i,:,:] * vis * np.cos(lats)) * dellat * dellon

    return integratedspec,integratederr,nearestint,fullplanetspec,fullplaneterr#eigenspectra,eigenlist

