# The input sph array will have the last two dimensions be (wavelengths,
# harmonic coefficients + 1). The first entry in each row is assumed to be
# the wavelength (in nm) corresponding to the spherical harmonic coefficients
# in that row. Note that the input array can have any dimensions prior to the
# last two.

# The returned array has dimensions (..., wavelengths, latitudes, longitudes),
# where the ellipses denote any extra dimensions from the input array.

import numpy as np
from scipy.special import sph_harm
import matplotlib.pyplot as plt
import pdb
from matplotlib import rc

def generate_maps(sph, N_lon, N_lat):
    '''
    Compute brightness map on a grid using spherical harmonic coefficients.

    Parameters
    ----------
    sph : array of spherical harmonic coefficients (axes: ..., wavelengths, SH coeffs)
    N_lon : int, number of gridpoints in longitude
    N_lat : int, number of gridpoints in latitude

    Returns
    -------
    wavelengths : array containing wavelength of each map
    lats : array of latitudes (radians)
    lons : array of longitudes (radians)
    fluxes : array of brightness maps at each wavelength, computed on grid
    '''

    wavelengths = sph[..., 0]
    harmonics = sph[..., 1:]
    degree = int(np.sqrt(np.shape(harmonics)[-1]))

    # The scipy spherical harmonic routine requires coordinates in polar form.
    las = np.linspace(0, np.pi, N_lat)
    los = np.linspace(0, 2*np.pi, N_lon)

    sph_l = np.concatenate([np.tile(l, l+1) for l in range(degree)])
    sph_m = np.concatenate([np.arange(l+1) for l in range(degree)])

    base_harmonics = sph_harm(np.tile(sph_m, (N_lon, N_lat, 1)).T,
                              np.tile(sph_l, (N_lon, N_lat, 1)).T,
                              *np.meshgrid(los, las))

    fluxes = np.sum([
                np.einsum('m...wvu,m->...wvu', np.array(
                        [np.einsum('...wx,xvu->...wvu',
                                   harmonics[..., l*(l+1)+np.array([m, -m])],
                                   np.array([base_harmonics[l*(l+1)//2+m].real,
                                             base_harmonics[l*(l+1)//2+m].imag
                                             ]))
                            for m in range(l+1)]),
                        [1] + l * [((-1)**l)*np.sqrt(2)])
                for l in range(degree)], axis=0)

    # Here we convert to (-pi, pi) in longitude, and (-pi/2, pi/2) in latitude,
    # and multiply by the factor that normalizes the harmonics.
    fluxes = 2*np.sqrt(np.pi) * \
        np.flip(np.roll(fluxes, N_lon//2, axis=-1), axis=-2)

    lons, lats = np.meshgrid(los-np.pi, las-np.pi/2)
    fluxes=np.flip(fluxes,axis=1)

    return wavelengths, lats, lons, fluxes

def show_group_histos(savedir,numsamp,ngroups,extent,
                      xLons=[-0.5,-0.5,0.5, 0.5],
                      xLats=[-0.5, 0.5,0.5,-0.5]):
    """
    Show histograms of the groups for specific regions of the map
    
    Parameters
    ----------
    input_map: 2D numpy array
        Global map of brightness or another quantity
        Latitudes x Longitudes
    lats: 2D numpy array
        Latitudes for the input_map grid in radians
    lons: 2D numpy array
        Longitudes for the input_map grid in radians
    kgroup_draws: 3D numpy array
        Kgroup draws from k Means Nsamples x Latitudes? x Longitudes?
    
    xLons: 4 element list or numpy array
        longitudes of points to show histograms in radians
    xLats: 4 element list or numpy array
        latitudes of points to show histograms in radians
    input_map_units: str
        Label for the global map units
    saveName: str or None
        Name of plot to save
    alreadyTrimmed: bool
        Is the input map already trimmed to the dayside?
        If false, it will trim the global map
    lonsTrimmed: bool
        Is the longitude array already trimmed?
        If false, it will trim the longitude array
    """
    file=np.load(savedir+'eigenspectra_{}_draws_{}_groups.npz'.format(numsamp,ngroups),allow_pickle=True)
    kgroup_draws=file['arr_1']

    file2=np.load(savedir+'kgroups_{}_draws_{}_groups.npz'.format(numsamp,ngroups),allow_pickle=True)
    input_map=file2['arr_0']
    lats=file2['arr_1']
    lons=file2['arr_2']
    
    londim = input_map.shape[1]
    windowLabels = ['A','B','C','D']
    plotextent = np.array([np.min(lons[0,:])/np.pi*180,np.max(lons[0,:])/np.pi*180,np.min(lats[:,0])/np.pi*180,np.max(lats[:,0])/np.pi*180])
    
    rc('axes',linewidth=2)
    plt.figure()
    plotData = plt.imshow(input_map, extent=plotextent)
    cbar = plt.colorbar(plotData)
    cbar.set_label('Mean Group',fontsize=20)
    cbar.ax.tick_params(labelsize=15,width=2,length=6)
    for ind in np.arange(len(xLons)):
        xLon, xLat = xLons[ind], xLats[ind]
        iLon, iLat = np.argmin(np.abs(lons[0,:] - xLon)), np.argmin(np.abs(lats[:,0] - xLat))
        plt.text(lons[0,iLon]* 180./np.pi,lats[iLat,0]* 180./np.pi,windowLabels[ind],
                color='red')
    plt.ylabel('Latitude',fontsize=20)
    plt.xlabel('Longitude',fontsize=20)
    plt.tick_params(labelsize=20,width=2,length=8)
    plt.savefig(savedir+'meangroup_{}_draws_{}_groups.pdf'.format(numsamp,ngroups))
    plt.show()
    
    fig,axs=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True)
    for ind in np.arange(len(xLons)):
        xLon, xLat = xLons[ind], xLats[ind]
        # left, bottom, width, height = [windowLocationsX[ind], windowLocationsY[ind], 0.2, 0.2]
        # ax2 = fig.add_axes([left, bottom, width, height])
        iLon, iLat = np.argmin(np.abs(lons[0,:] - xLon)), np.argmin(np.abs(lats[:,0] - xLat))
        p1=int(ind/2)
        p2=int(ind-p1*2)
        axs[p1][p2].hist(kgroup_draws[:,iLat,iLon])
        axs[p1][p2].set_title(windowLabels[ind],fontsize=20)
        axs[p1][p2].set_xlim(-0.5,np.max(kgroup_draws) + 0.5)
        # axs[p1][p2].set_xlabel('Group',fontsize=20)
        # axs[p1][p2].set_ylabel('# of Samples',fontsize=20)
        axs[p1][p2].tick_params(labelsize=20,axis="both",top=True,right=True,width=2,length=8,direction='in')
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel("Group",fontsize=20)
    plt.ylabel("# of Samples",fontsize=20,labelpad=30)
    plt.savefig(savedir+'histos_{}_draws_{}_groups.pdf'.format(numsamp,ngroups))
    plt.show()

    return

