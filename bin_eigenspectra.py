import numpy as np

def bin_eigenspectra(maps, kgroup_draws, lats, lons, ngroups):
    '''
    Converts a grid of spectra into eigenspectra defined by kgroups.

    Parameters
    ----------
    maps : array of retrieved maps, units scaled Fp/Fs 
        (axes: number of samples from fitting x wavelengths x latitude x longitude)
    kgroup_draws : grouping for each retrieved map, unitless 
        (axes: number of samples from fitting x latitude x longitude)
    lats : array of latitude, units radians (axes: latitude x longitude)
    lons : array of longitude, units radians (axes: latitude x longitude)
    ngroups : number of groups to bin into, unitless

    Returns
    -------
    integratedspec : spectra of each identified group, scaled to a full eclipse depth
        as if that spectrum covered an entire dayside hemisphere. Units: Fp/Fs
        (axes: number of groups x wavelength)
    integratederr : error of each identified group, scaled to a full eclipse depth
        as if that spectrum covered an entire dayside hemisphere. Units: Fp/Fs
        (axes: number of groups x wavelength)
    nearestint : array identifying the group each point was sorted into, unitless
        (axes: latitude x longitude)
    '''

    #Find integer group each point is sorted into most frequently
    kgroups = np.mean(kgroup_draws, axis=0)
    nearestint= np.round(kgroups,decimals=0)

    #Calculate mean and std of spectrum at each lat/long point
    perpointspec = np.mean(maps,axis=0)
    perpointerr = np.std(maps,axis=0)
    
    #Use an area weighting to get mean spectrum of each group
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

    #integrate over full sphere to get spectra as if observing a full planet covered
    #by that spectrum
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

    return integratedspec,integratederr,nearestint

