import numpy as np
import matplotlib.pyplot as plt

import eigenmaps
import kmeans
import bin_eigenspectra
import os
import pdb
import spiderman as sp

import time
import glob
import matplotlib.pylab as pl
import cmasher as cmr
import cartopy.crs as ccrs

def plot_setup():
    """
    Set some default plotting parameters
    """
    from matplotlib import rcParams
    rcParams["savefig.dpi"] = 200
    rcParams["figure.dpi"] = 100
    rcParams["font.size"] = 20
    rcParams["figure.figsize"] = [8, 5]
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = ["Computer Modern Sans Serif"]
    rcParams["text.usetex"] = True

def create_linear_colormap(c1 = "white", c2 = "C4", c3 = None, N = 1000, cmap_name = "custom_cmap"):
    """
    Creates a colormap with a linear gradient between two user-specified colors

    Parameters
    ----------
    c1 : str
        Color of the smallest value
    c2 : str
        Color of the largest/middle value
    c3 : str
        Color of the largest value
    N : int
        Color resolution
    cmap_name : str
        Name of new colormap

    Returns
    -------
    cm : matplotlib.colors.LinearSegmentedColormap
        New colormap
    """

    from matplotlib.colors import LinearSegmentedColormap, colorConverter

    # If a third color was not specified
    if c3 is None:

        # Create list with two end-member RGBA color tuples
        c = [colorConverter.to_rgba(c1), colorConverter.to_rgba(c2)]

    else:

        # Create list with two end-member RGBA color tuples
        c = [colorConverter.to_rgba(c1), colorConverter.to_rgba(c2), colorConverter.to_rgba(c3)]

    # Create the colormap
    cm = LinearSegmentedColormap.from_list(cmap_name, c, N = N)

    return cm

def retrieve_map_full_samples(planetparams,dataDir,outputpath,londim=360,latdim=180,\
    afew=3,degree=3,nrandom=100,isspider=True):
    if isinstance(degree,list):
        if isinstance(afew,int):
            if afew>=16:
                tmp = np.load("{}spherearray_deg_list_eigenfit.npz".format(dataDir,degree),allow_pickle=True)
            else:
                tmp = np.load("{}spherearray_deg_list_eigen_{}.npz".format(dataDir,degree,afew),allow_pickle=True)
        else:
            tmp = np.load("{}spherearray_deg_list.npz".format(dataDir,degree),allow_pickle=True)
    else:
        if isinstance(afew,int):
            if afew>=16:
                tmp = np.load("{}spherearray_deg_{}_eigenfit.npz".format(dataDir,degree),allow_pickle=True)
            else:
                tmp = np.load("{}spherearray_deg_{}_eigen_{}.npz".format(dataDir,degree,afew),allow_pickle=True)
        else:
            tmp = np.load("{}spherearray_deg_{}.npz".format(dataDir,degree),allow_pickle=True)

    # outputpath='./data/test/'
    if os.path.exists(outputpath) == False:
        os.makedirs(outputpath)
    outDictionary = tmp['arr_0'].tolist()

    waves = outDictionary['wavelength (um)']
    eigensamples = outDictionary['eigencurve coefficients'] # output from eigencurves
    bestsamples=outDictionary['best fit coefficients'] # best fit sample from eigencurves #FINDME: REPLACE WITH 'best fit spherical coefficients'
    ecoeffList=outDictionary['ecoeffList']
    # londim = 360
    # latdim = 180
    # nRandom= nrandom#500
    if nrandom=='all':
        nRandom=[]
        for i in np.arange(np.shape(waves)[0]):
            nRandom.append(len(eigensamples[i]))
    else:
        nRandom=nrandom*np.ones(np.shape(waves)[0])

    # samples = []
    # for waveind in np.arange(np.shape(waves)[0]):
        

    # pdb.set_trace()
    #np.random.randint(0,len(samples),100) #FINDME: CHANGED FROM 1000 and also changed from samples to samples[0] for degree indexing
    # print(randomIndices)
    # pdb.set_trace()
    # if nRandom=='all':
    #     fullMapArray=[]
    #     for i in np.arange(len(waves)):
    #         fullMapArray.append(np.zeros([nRandom,latdim,londim]))
    # else:
    fullMapArray = []
    fcorrfull = []
    for i in np.arange(len(waves)):
        fullMapArray.append(np.zeros([int(nRandom[i]),latdim,londim]))#londim,latdim])
        fcorrfull.append(np.zeros(int(nRandom[i])))
    bestMapArray = np.zeros([len(waves),latdim,londim])#londim,latdim])
    bestDepths = np.zeros(len(waves))
    fcorrbest = np.zeros(len(waves))
    for waveind in np.arange(len(waves)):
        savefile=outputpath+'wave_{:0.2f}'.format(waves[waveind])
        # pdb.set_trace()
        if os.path.exists(savefile+'.npz'):
            print('Found the previously saved file {}. Now exiting.'.format(savefile))
        else:
            print('Converting to spherical harmonics for wave={:0.2f}.'.format(waves[waveind]))
            # counter=0
            # if nRandom=='all':
            #     randomIndices = np.arange(len(eigensamples[waveind]))
            # else:
            randomIndices = np.random.choice(len(eigensamples[waveind]),int(nRandom[waveind]),replace=False)
            tempsamp=eigensamples[waveind]
            ecoeff=ecoeffList[waveind]
            # if nRandom=='all':
            #     samparray=np.zeros((len(eigensamples[waveind]),int((degree[waveind])**2.)))
            # else:
            samparray=np.zeros((int(nRandom[waveind]),int((degree[waveind])**2.)))
            counter=0
            for sampnum in randomIndices: #THIS IS THE PART THAT TAKES FOREVER
                fcoeff=np.zeros_like(ecoeff)
                # newfcoeff=np.zeros_like(ecoeff)
                # for i in np.arange(np.shape(tempsamp)[1]-2):
                #     fcoeff[:,i] = tempsamp[sampnum,i+2]*ecoeff[:,i]
                tfcoeff=tempsamp[sampnum, 2:] * ecoeff[:, :(np.shape(tempsamp)[1]-2)]
                fcoeff[:,:(np.shape(tempsamp)[1]-2)]=tfcoeff
                # how to go from coefficients to best fit map
                spheres=np.zeros(int((degree[waveind])**2.))
                # tspheres=np.zeros(int((degree[waveind])**2.))
                for j in range(len(fcoeff)):
                    # for i in range(1,int((degree[waveind])**2.)):
                    #     spheres[i] += np.real(fcoeff.T[j,2*i-1]-fcoeff.T[j,2*(i-1)])
                    sphere_update = np.real(fcoeff.T[j,1::2] - fcoeff.T[j,::2])
                    spheres[1:] += sphere_update

                spheres[0] = tempsamp[sampnum,0]#bestcoeffs[0]#c0_best
                samparray[counter,:]=spheres
                fcorrfull[waveind][counter] = eigensamples[waveind][sampnum,1]
                counter+=1
            # samples.append(samparray)
            # pdb.set_trace()
            fcorrbest[waveind]=np.median(eigensamples[waveind][:,1])

            params0=sp.ModelParams(brightness_model='spherical')
            params0.nlayers=20
            params0.t0=planetparams['t0']
            params0.per=planetparams['per']
            params0.a_abs=planetparams['a_abs']
            params0.inc=planetparams['inc']
            params0.ecc=planetparams['ecc']
            params0.w=planetparams['w']
            params0.rp=planetparams['rprs']
            params0.a=planetparams['ars']
            params0.p_u1=0.
            params0.p_u2=0.
            params0.la0=0.
            params0.lo0=0.
            params0.degree=degree[waveind]
            params0.sph=list(bestsamples[waveind])
            bestDepths[waveind] = params0.eclipse_depth()
            #SPIDERMAN stuff added by Megan
            if isspider:
                # for inddeg in np.arange(np.min(degree),np.max(degree)+1):
                #     params0.degree=inddeg
                #     lookatme=np.where(degree==inddeg)[0]
                #     for indwave in lookatme:
                nla=latdim
                nlo=londim
                las = np.linspace(-np.pi/2,np.pi/2,nla)
                los = np.linspace(-np.pi,np.pi,nlo)
                fluxes = []
                for la in las:
                    row = []
                    for lo in los:
                        flux = sp.call_map_model(params0,la,lo)
                        row += [flux[0]]
                    fluxes += [row]
                fluxes = np.array(fluxes)
                lons, lats = np.meshgrid(los,las)
                newfluxes = normalize_flux(fluxes,lats,lons,bestDepths[waveind],fcorrbest[waveind]) #FINDME
                bestMapArray[waveind,:,:] = newfluxes #FINDME
                
            if not isspider:
                inputArr=np.zeros([len(waves),int(degree[waveind]**2+1)])
                # for inddeg in np.arange(np.min(degree),np.max(degree)+1):
                #     lookatme=np.where(degree==inddeg)[0]
                #     inputArr=np.zeros([len(waves),int(inddeg**2+1)])
                inputArr[:,0] = waves
                    # for indwave in lookatme:
                inputArr[waveind,1:] = bestsamples[waveind].transpose()
                wavelengths, lats, lons, maps = eigenmaps.generate_maps(inputArr,N_lon=londim, N_lat=latdim)
                t0=time.time()
                newmaps = normalize_flux(maps[waveind],lats,lons,bestDepths[waveind],fcorrbest[waveind]) #FINDME
                t1=time.time()
                print(t1-t0)
                bestMapArray[waveind]=newmaps#maps[waveind] #FINDME
        # for i in np.arange(np.shape(waves)[0]):
            print('Converting to maps for wave={:0.2f}.'.format(waves[waveind]))
            # counter=0
            t0=time.time()
            for drawInd in np.arange(nRandom[waveind]):
                fcorr = fcorrfull[waveind][drawInd]
                if not isspider:
                    inddeg=degree[waveind]
                    inputArr=np.zeros([len(waves),int(inddeg**2+1)])
                    inputArr[:,0] = waves
                    inputArr[:,1:] = samparray[int(drawInd),:].transpose()
                    wavelengths, lats, lons, maps = eigenmaps.generate_maps(inputArr,
                                                                            N_lon=londim, N_lat=latdim)
                    newmaps = normalize_flux(maps[waveind],lats,lons,bestDepths[waveind],fcorr) #FINDME
                    fullMapArray[waveind][int(drawInd),:,:] = newmaps #maps[waveind] FINDME

                else:
                    inddeg=degree[waveind]
                    params0.degree=inddeg
                    params0.sph=list(samparray[int(drawInd),:])
                    nla=latdim
                    nlo=londim
                    las = np.linspace(-np.pi/2,np.pi/2,nla)
                    los = np.linspace(-np.pi,np.pi,nlo)
                    fluxes = []
                    for la in las:
                        row = []
                        for lo in los:
                            flux = sp.call_map_model(params0,la,lo)
                            row += [flux[0]]
                        fluxes += [row]
                    fluxes = np.array(fluxes)
                    lons, lats = np.meshgrid(los,las)
                    newmaps = normalize_flux(fluxes,lats,lons,bestDepths[waveind],fcorr) #FINDME
                    fullMapArray[waveind][int(drawInd),:,:] = newmaps#fluxes #FINDME
            t1=time.time()
            print(nRandom[waveind],t1-t0)
            np.savez(savefile,waves[waveind],lats,lons,bestMapArray[waveind],fullMapArray[waveind],bestDepths[waveind],fcorrbest[waveind],fcorrfull[waveind])

    return outputpath#fullMapArray, bestMapArray, lats, lons, waves, bestDepths, fcorr

def normalize_flux(tmap,lats,lons,depth,fcorr):
    centlon=0.
    centlat=0.
    dellat=np.diff(lats[:,0])[0]
    dellon=np.diff(lons[0,:])[0]
    centlat_sin = np.sin(centlat)
    centlat_cos = np.cos(centlat)

    # Compute vis using vectorized operations
    vis = centlat_sin * np.sin(lats) + centlat_cos * np.cos(lats) * \
            np.cos(lons - centlon)
    vis[vis <= 0.] = 0.
    theintegral = np.sum(tmap * vis * np.cos(lats)) * dellat * dellon

    return tmap/theintegral*depth*fcorr

def calc_minmax(datadir,extent):
    print('Calculating minimum and maximum flux.')
    flist=glob.glob(os.path.join(datadir,'wave*.npz'))
    percentiles = [16,50,84]
    minflux=0.
    maxflux=0.
    counter=0
    for f in flist:
        print('Step '+str(counter))
        alldict=np.load(f,allow_pickle=True)
        fullMapArray=alldict['arr_4']
        bestMapArray=alldict['arr_3']
        lons=alldict['arr_2']
        lats=alldict['arr_1']
        goodplot=np.where((lons[0,:]>extent[0])&(lons[0,:]<extent[1]))[0]
        mapLowMedHigh=np.percentile(fullMapArray,percentiles,axis=0)
        mapLowMedHigh[1,:,:]=bestMapArray
        tminflux=np.min(mapLowMedHigh[:,:,np.min(goodplot):np.max(goodplot)+1])
        tmaxflux=np.max(mapLowMedHigh[:,:,np.min(goodplot):np.max(goodplot)+1])
        if (minflux==0.) | (tminflux<minflux):
            minflux=tminflux
        if (maxflux==0.) | (tmaxflux>maxflux):
            maxflux=tmaxflux
        counter+=1

    return minflux,maxflux

def plot_retrieved_map(datadir,waves,extent,minflux,maxflux,waveInd=3,saveName=None):

    flist=glob.glob(os.path.join(datadir,'wave*.npz'))
    percentiles = [16,50,84]


    if os.path.exists(datadir+'/retrieved_maps/') == False:
        os.makedirs(datadir+'/retrieved_maps/')

    if isinstance(waveInd,int):
        wavesround=np.round(waves[waveInd],decimals=2)
        for f in flist:
            if float(f[-8:-4])==wavesround:
                goodf=f 
        alldict=np.load(goodf,allow_pickle=True)
        fullMapArray=alldict['arr_4']
        bestMapArray=alldict['arr_3']
        lons=alldict['arr_2']
        lats=alldict['arr_1']
        goodplot=np.where((lons[0,:]>extent[0])&(lons[0,:]<extent[1]))[0]
        mapLowMedHigh=np.percentile(fullMapArray,percentiles,axis=0)
        mapLowMedHigh[1,:,:]=bestMapArray
        rc('axes',linewidth=2)
        fig=plt.figure(figsize=(10,6.5))
        ax=plt.axes([0.1,0.35,0.8,0.6],projection=ccrs.Robinson())
        # plt.figure()
        map_day = mapLowMedHigh[1][:,goodplot]
        plotextent = np.array([np.min(lons[0,goodplot])/np.pi*180,np.max(lons[0,goodplot])/np.pi*180,np.min(lats[:,0])/np.pi*180,np.max(lats[:,0])/np.pi*180])
        # plotData = plt.imshow(map_day, extent=plotextent,vmin=minflux,vmax=maxflux)
        vis = calc_contribution(lats[:,goodplot],lons[:,goodplot])
        plotData = ax.imshow(map_day, extent=plotextent,vmin=minflux,vmax=maxflux, alpha=vis, transform=ccrs.PlateCarree(),cmap='inferno')
        # cbar = plt.colorbar(plotData)
        plt.title('Wavelength={:0.2f} $\mu$m'.format(waves[waveInd]),fontsize=15)
        gl=ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True)
        gl.xlabel_style = {'size': 15, 'color': 'k'}
        gl.ylabel_style = {'size': 15, 'color': 'k'}
        # plt.ylabel('Latitude',fontsize=20)
        # plt.xlabel('Longitude',fontsize=20)
        rc('axes',linewidth=1)
        a = plt.axes([0.2,0.1,0.6,0.15])
        # a.yaxis.set_visible(False)
        # a.xaxis.set_visible(False)
        colorbardata=np.zeros((100,100))
        colorbaralpha=np.zeros((100,100))
        for i in np.arange(100):
            colorbardata[i,:]=np.linspace(minflux*10**6.,maxflux*10**6.,100)
            colorbaralpha[:,i]=np.linspace(0,1,100)
        cbarextent=np.array([minflux*10**6.,maxflux*10**6.,1.,0.])
        astuff = a.imshow(colorbardata, alpha=colorbaralpha,aspect='auto',extent=cbarextent,cmap='inferno')#, visible=False)
        a.set_xlabel('Fp/Fs [ppm]',fontsize=15)
        a.set_ylabel('Contribution',fontsize=15)
        a.tick_params(labelsize=15,axis="both",top=True,right=True,width=1,length=4,direction='in')
        # cbar=plt.colorbar(a, fraction=3.0,orientation='horizontal')
        # cbar.ax.tick_params(labelsize=15,width=2,length=6)
        # cbar.set_label('Test',fontsize=15)
        # pdb.set_trace()
        # cbar.set_label('Fp/Fs [ppm]',fontsize=15)
        # cbar.ax.tick_params(labelsize=15,width=2,length=6)
        # plt.ylabel('Latitude',fontsize=20)
        # plt.xlabel('Longitude',fontsize=20)
        # pdb.set_trace()
        # plt.tick_params(labelsize=20,axis="both",top=True,right=True,width=2,length=8,direction='in')
        #axArr[ind].set_title("{} %".format(onePercentile))
        #axArr[ind].show()
        # plt.title('{:.2f}$\mu$m'.format(waves[waveInd]),fontsize=20)
        # plt.tight_layout()
        plt.savefig(datadir+'/retrieved_maps/retrieved_map_{}_wave_{}.pdf'.format(saveName,wavesround))
        plt.show()
        # pdb.set_trace()

        fig, axArr = plt.subplots(1,3,figsize=(22,5))
        for ind,onePercentile in enumerate(percentiles):
            map_day = mapLowMedHigh[ind][:,goodplot]
            plotData = axArr[ind].imshow(map_day, extent=plotextent,vmin=minflux,vmax=maxflux,cmap='inferno')
            axArr[ind].set_ylabel('Latitude')
            axArr[ind].set_xlabel('Longitude')
            axArr[ind].set_title("{} %".format(onePercentile))
        fig.suptitle('{:.2f}$\mu$m'.format(singlewave),fontsize=20)
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(plotData, cax=cbar_ax)
        cbar.set_label('Brightness')
        plt.savefig(datadir+'/retrieved_maps/retrieved_3_maps_{}_wave_{}.pdf'.format(saveName,wavesround))
        plt.show()

    elif waveInd=='Full':
        for f in flist:
            alldict=np.load(f,allow_pickle=True)
            fullMapArray=alldict['arr_4']
            bestMapArray=alldict['arr_3']
            lons=alldict['arr_2']
            lats=alldict['arr_1']
            singlewave=np.round(alldict['arr_0'],decimals=2)
            goodplot=np.where((lons[0,:]>extent[0])&(lons[0,:]<extent[1]))[0]
            mapLowMedHigh=np.percentile(fullMapArray,percentiles,axis=0)
            mapLowMedHigh[1,:,:]=bestMapArray
            # rc('axes',linewidth=2)
            # plt.figure()
            rc('axes',linewidth=2)
            fig=plt.figure(figsize=(10,6.5))
            ax=plt.axes([0.1,0.35,0.8,0.6],projection=ccrs.Robinson())  
            map_day = mapLowMedHigh[1][:,goodplot]
            plotextent = np.array([np.min(lons[0,goodplot])/np.pi*180,np.max(lons[0,goodplot])/np.pi*180,np.min(lats[:,0])/np.pi*180,np.max(lats[:,0])/np.pi*180])
            vis = calc_contribution(lats[:,goodplot],lons[:,goodplot])
            plotData = ax.imshow(map_day, extent=plotextent,vmin=minflux,vmax=maxflux, alpha=vis, transform=ccrs.PlateCarree(),cmap='inferno')
            # plotData = plt.imshow(map_day, extent=plotextent,vmin=minflux,vmax=maxflux)
            plt.title('Wavelength={:0.2f} $\mu$m'.format(singlewave),fontsize=15)
            gl=ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True)
            gl.xlabel_style = {'size': 15, 'color': 'k'}
            gl.ylabel_style = {'size': 15, 'color': 'k'}

            rc('axes',linewidth=1)
            a = plt.axes([0.2,0.1,0.6,0.15])
            colorbardata=np.zeros((100,100))
            colorbaralpha=np.zeros((100,100))
            for i in np.arange(100):
                colorbardata[i,:]=np.linspace(minflux*10**6.,maxflux*10**6.,100)
                colorbaralpha[:,i]=np.linspace(0,1,100)
            cbarextent=np.array([minflux*10**6.,maxflux*10**6.,1.,0.])
            astuff = a.imshow(colorbardata, alpha=colorbaralpha,aspect='auto',extent=cbarextent,cmap='inferno')#, visible=False)
            a.set_xlabel('Fp/Fs [ppm]',fontsize=15)
            a.set_ylabel('Contribution',fontsize=15)
            a.tick_params(labelsize=15,axis="both",top=True,right=True,width=1,length=4,direction='in')
            # cbar = plt.colorbar(plotData)
            # cbar.set_label('Fp/Fs',fontsize=15)
            # cbar.ax.tick_params(labelsize=15,width=2,length=6)
            # plt.ylabel('Latitude',fontsize=20)
            # plt.xlabel('Longitude',fontsize=20)
            # plt.tick_params(labelsize=20,axis="both",top=True,right=True,width=2,length=8,direction='in')
            #axArr[ind].set_title("{} %".format(onePercentile))
            #axArr[ind].show()
            # plt.title('{}$\mu$m'.format(singlewave),fontsize=20)
            # plt.tight_layout()
            plt.savefig(datadir+'/retrieved_maps/retrieved_map_{}_wave_{}.pdf'.format(saveName,singlewave))
            plt.show()

            fig, axArr = plt.subplots(1,3,figsize=(22,5))
            for ind,onePercentile in enumerate(percentiles):
                map_day = mapLowMedHigh[ind][:,goodplot]
                plotData = axArr[ind].imshow(map_day, extent=plotextent,vmin=minflux,vmax=maxflux,cmap='inferno')
                axArr[ind].set_ylabel('Latitude')
                axArr[ind].set_xlabel('Longitude')
                axArr[ind].set_title("{} %".format(onePercentile))
            fig.suptitle('{:.2f}$\mu$m'.format(singlewave),fontsize=20)
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(plotData, cax=cbar_ax)
            cbar.set_label('Brightness')
            plt.savefig(datadir+'/retrieved_maps/retrieved_3_maps_{}_wave_{}.pdf'.format(saveName,singlewave))
            plt.show()

    return mapLowMedHigh#, mintemp, maxtemp

def calc_contribution(lats,lons):
    centlon=0.
    centlat=0.
    dellat=np.diff(lats[:,0])[0]
    dellon=np.diff(lons[0,:])[0]
    centlat_sin = np.sin(centlat)
    centlat_cos = np.cos(centlat)

    # Compute vis using vectorized operations
    vis = centlat_sin * np.sin(lats) + centlat_cos * np.cos(lats) * \
            np.cos(lons - centlon) * np.cos(lats)
    vis[vis <= 0.] = 0.

    return vis

def calc_contribution_allang(lats,lons,mincentlon,maxcentlon):
    #calculate at minimum and maximum central longitude for the eclipse
    centlon=0.
    centlat=0.
    dellat=np.diff(lats[:,0])[0]
    dellon=np.diff(lons[0,:])[0]
    centlat_sin = np.sin(centlat)
    centlat_cos = np.cos(centlat)

    # Compute vis using vectorized operations
    vis1 = centlat_sin * np.sin(lats) + centlat_cos * np.cos(lats) * \
            np.cos(lons - mincentlon) * np.cos(lats)
    vis2 = centlat_sin * np.sin(lats) + centlat_cos * np.cos(lats) * \
            np.cos(lons - maxcentlon) * np.cos(lats)
    vismax = centlat_sin * np.sin(lats[:,0]) + centlat_cos * np.cos(lats[:,0]) * \
            np.cos(0. - centlon) * np.cos(lats[:,0])
    # pdb.set_trace()
    vis=np.zeros(np.shape(vis1))
    vis[np.where(vis1>vis2)]=vis1[np.where(vis1>vis2)]
    vis[np.where(vis2>vis1)]=vis2[np.where(vis2>vis1)]
    minindex=np.argmin(abs(mincentlon-lons[0,:]))
    maxindex=np.argmin(abs(maxcentlon-lons[0,:]))
    for index in np.arange(minindex,maxindex+1):
        vis[:,index] = vismax
    vis[vis <= 0.] = 0.

    return vis
# def norm_maps_for_temp(outputpath,savepath,extent):
#     flist=glob.glob(os.path.join(outputpath,'*.npz'))
#     percentiles = [5,50,95]

#     if os.path.exists(savepath) == False:
#         os.makedirs(savepath)

#     counter=0
#     for f in flist:
#         print('Step '+str(counter))
#         alldict=np.load(f,allow_pickle=True)
#         fullMapArray=alldict['arr_4']
#         bestMapArray=alldict['arr_3']
#         lons=alldict['arr_2']
#         lats=alldict['arr_1']
#         wave=alldict['arr_0']
#         bestDepth=alldict['arr_5']
#         fcorr=alldict['arr_6']
#         goodplot=np.where((lons[0,:]>extent[0])&(lons[0,:]<extent[1]))[0]
#         mapLowMedHigh=np.percentile(fullMapArray,percentiles,axis=0)
#         mapLowMedHigh[1,:,:]=bestMapArray
#         newmedianmap=np.zeros(np.shape(mapLowMedHigh))
#         for percentage in np.arange(len(percentiles)):
#             medianmap=mapLowMedHigh[percentage]
#             centlon=0.
#             centlat=0.
#             theintegral=0.
#             dellat=np.diff(lats[:,0])[0]
#             dellon=np.diff(lons[0,:])[0]

#             for l in np.arange(np.shape(medianmap)[0]):
#                 for m in np.arange(np.shape(medianmap)[1]):
#                     vis=np.sin(centlat)*np.sin(lats[l,m])+np.cos(centlat)*np.cos(lats[l,m])*np.cos(lons[l,m]-centlon)
#                     if vis>0.:
#                         theintegral += medianmap[l,m]*vis*np.cos(lats[l,m])*dellat*dellon
#             newmedianmap[percentage,:,:]=medianmap[:,:]/theintegral

#         savefile=savepath+'wave_{:0.2f}'.format(wave)
#         np.savez(savefile,wave,lats,lons,newmedianmap,bestDepth,fcorr)

#         counter+=1
#     return savepath

def specint(wavelength,spec,throughput):
    normthrough=throughput/np.trapz(throughput,wavelength)
    integrated=np.trapz(spec*normthrough,wavelength)
    return integrated

def planetbbod(T,y,lam,throughput):
    c=2.998*10**8.
    h=6.626*10**-34.
    kb=1.381*10**-23.
    planet=2*h*c**2./lam**5./(np.exp(h*c/(lam*kb*T))-1)
    planetint=specint(lam,planet,throughput)
    return np.array(y-planetint)

def convert_temp(wavegrid,centwave,inputmap,rprs,throughput,startype='Temp',Tstar=6000.,smodel=None): #fpfs,fcorr,#wavegrid in microns
    from scipy.optimize import leastsq
    c=2.998*10**8.
    h=6.626*10**-34.
    kb=1.381*10**-23.
    wavesinm=wavegrid*10**-6.
    if startype=='Temp':
        starbb=2*h*c**2./wavesinm**5./(np.exp(h*c/(wavesinm*kb*Tstar))-1)
        starint=specint(wavesinm,starbb,throughput)
    elif startype=='Model':
        #assume that model is in correct units (wavelengths in m and flux in W/m^3/sr)
        starbb = np.interp(wavesinm,smodel[:,0],smodel[:,1])
        starint=specint(wavesinm,starbb,throughput)
    centwave*=10**-6.
    # ptemp=h*c/(centwave*kb) #FINDME: THIS IS THE LAZY WAY
    # mapintemp1=np.zeros(np.shape(inputmap))
    # for i in np.arange(np.shape(inputmap)[0]):
    #     for j in np.arange(np.shape(inputmap)[1]):
    #         mapintemp1[i,j]=ptemp/np.log(1+rprs**2.*(2*h*c**2.)\
    #             /(centwave**5.*np.pi*inputmap[i,j]*fcorr*starint*fpfs))
    mapintemp=np.zeros(np.shape(inputmap))
    for i in np.arange(np.shape(inputmap)[0]):
        for j in np.arange(np.shape(inputmap)[1]):
            theflux=np.pi*starint*inputmap[i,j]/rprs**2.#fpfs*fcorr
            temp0=np.array([1000.])
            tfit=leastsq(planetbbod,temp0,args=(theflux,wavesinm,throughput))
            mapintemp[i,j]=tfit[0]
    return mapintemp


def plot_map_in_temp(datadir,savepath,waves,dlam,rprs,extent,waveInd=0,saveName=None,startype='Temp',Tstar=6000.,smodel=None):
    #FINDME: make these plots in the nice format
    percentiles = [16,50,84]
    flist=glob.glob(os.path.join(datadir,"wave*.npz"))
    assert (startype=='Temp') | (startype=='Model'), "Only startype='Temp' and startype='Model' are supported options."

    mincentlon=extent[0]+np.pi/2. #minimum subobserver point, in radians
    maxcentlon=extent[1]-np.pi/2.

    if os.path.exists(savepath) == False:
        os.makedirs(savepath)
    if os.path.exists(savepath+'/retrieved_maps/') == False:
        os.makedirs(savepath+'/retrieved_maps/')

    if isinstance(waveInd,int):
        wavesround=np.round(waves[waveInd],decimals=2)
        if startype=='Temp':
            savefile=savepath+'temp_wave_Teff={}_{:0.2f}'.format(int(Tstar),waves[waveInd])
        elif startype=='Model':
            savefile=savepath+'temp_wave_Model_{:0.2f}'.format(waves[waveInd])
        if os.path.exists(savefile+'.npz'):
            print('Found the previously saved file {}. Now exiting.'.format(savefile))
            prevfile=np.load(savefile+'.npz')
            lats=prevfile['arr_1']
            lons=prevfile['arr_2']
            mapintemp=prevfile['arr_3']


        else:
            for f in flist:
                if float(f[-8:-4])==wavesround:
                    goodf=f 
            alldict=np.load(goodf,allow_pickle=True)
            fullMapArray=alldict['arr_4']
            bestMapArray=alldict['arr_3']
            lons=alldict['arr_2']
            lats=alldict['arr_1']
            inputmap=np.percentile(fullMapArray,percentiles,axis=0)
            inputmap[1,:,:]=bestMapArray
            mapintemp=np.zeros(np.shape(inputmap))
            inputmap[inputmap<0.]=0.
            for medindex in np.arange(np.shape(mapintemp)[0]):
                wavegrid=np.linspace(waves[waveInd]-dlam[waveInd]/2.,waves[waveInd]+dlam[waveInd]/2.,1000)
                throughput=calc_instrument_throughput('NIRISS',wavegrid)
                if startype=='Temp':
                    mapintemp[medindex]=convert_temp(wavegrid,waves[waveInd],inputmap[medindex],rprs,throughput,startype=startype,Tstar=Tstar)#fpfs,fcorr,
                elif startype=='Model':
                    mapintemp[medindex]=convert_temp(wavegrid,waves[waveInd],inputmap[medindex],rprs,throughput,startype=startype,smodel=smodel)
            
            np.savez(savefile,alldict['arr_0'],lats,lons,mapintemp)

        goodplot=np.where((lons[0,:]>extent[0])&(lons[0,:]<extent[1]))[0]
        maxtemp=np.max(mapintemp[:,:,np.min(goodplot):np.max(goodplot)+1])
        mintemp=np.min(mapintemp[:,:,np.min(goodplot):np.max(goodplot)+1])

        rc('axes',linewidth=2)
        fig=plt.figure(figsize=(10,6.5))
        ax=plt.axes([0.1,0.35,0.8,0.6],projection=ccrs.Robinson()) 
        plotextent = np.array([np.min(lons[0,goodplot])/np.pi*180,np.max(lons[0,goodplot])/np.pi*180,np.min(lats[:,0])/np.pi*180,np.max(lats[:,0])/np.pi*180])
        vis = calc_contribution_allang(lats[:,goodplot],lons[:,goodplot],mincentlon,maxcentlon)
        plotData = ax.imshow(mapintemp[1][:,goodplot], extent=plotextent,vmin=mintemp,vmax=maxtemp, alpha=vis, transform=ccrs.PlateCarree(),cmap='inferno')
        # plotData = plt.imshow(mapintemp[1][:,goodplot], extent=plotextent,vmin=mintemp,vmax=maxtemp)
        # cbar = plt.colorbar(plotData)
        plt.title('Wavelength={} $\mu$m'.format(wavesround),fontsize=15)
        gl=ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True)
        gl.xlabel_style = {'size': 15, 'color': 'k'}
        gl.ylabel_style = {'size': 15, 'color': 'k'}

        rc('axes',linewidth=1)
        a = plt.axes([0.2,0.1,0.6,0.15])
        colorbardata=np.zeros((100,100))
        colorbaralpha=np.zeros((100,100))
        for i in np.arange(100):
            colorbardata[i,:]=np.linspace(mintemp,maxtemp,100)
            colorbaralpha[:,i]=np.linspace(0,1,100)
        cbarextent=np.array([mintemp,maxtemp,1.,0.])
        astuff = a.imshow(colorbardata, alpha=colorbaralpha,aspect='auto',extent=cbarextent,cmap='inferno')#, visible=False)
        a.set_xlabel('Temperature [K]',fontsize=15)
        a.set_ylabel('Contribution',fontsize=15)
        a.tick_params(labelsize=15,axis="both",top=True,right=True,width=1,length=4,direction='in')
        # cbar.set_label('Temperature [K]',fontsize=15)
        # cbar.ax.tick_params(labelsize=15,width=2,length=6)
        # plt.ylabel('Latitude',fontsize=20)
        # plt.xlabel('Longitude',fontsize=20)
        # plt.tick_params(labelsize=20,axis="both",top=True,right=True,width=2,length=8,direction='in')
        # plt.title('{}$\mu$m'.format(wavesround),fontsize=20)
        # plt.tight_layout()
        if startype=='Temp':
            plt.savefig(savepath+'/retrieved_maps/tempmap_{}_Teff={}_wave_{:0.2f}.pdf'.format(saveName,int(Tstar),waves[waveInd]))
        elif startype=='Model':
            plt.savefig(savepath+'/retrieved_maps/tempmap_{}_Model_wave_{:0.2f}.pdf'.format(saveName,waves[waveInd]))
        plt.show()

        fig, axArr = plt.subplots(1,3,figsize=(22,5))
        for ind,onePercentile in enumerate(percentiles):
            plotData = axArr[ind].imshow(mapintemp[ind][:,goodplot], extent=plotextent,vmin=mintemp,vmax=maxtemp,cmap='inferno')
            axArr[ind].set_ylabel('Latitude')
            axArr[ind].set_xlabel('Longitude')
            axArr[ind].set_title("{} %".format(onePercentile))
        fig.suptitle('{:.2f}$\mu$m'.format(wavesround),fontsize=20)
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(plotData, cax=cbar_ax)
        cbar.set_label('Temperature [K]')
        if startype=='Temp':
            plt.savefig(savepath+'/retrieved_maps/3_tempmaps_{}_Teff={}_wave_{:0.2f}.pdf'.format(saveName,int(Tstar),waves[waveInd]))
        elif startype=='Model':
            plt.savefig(savepath+'/retrieved_maps/3_tempmaps_{}_Model_wave_{:0.2f}.pdf'.format(saveName,waves[waveInd]))
        plt.show()

    elif waveInd=='Full':
        maxtemp=1500.
        mintemp=1500.
        counter=0
        for f in flist:
            print('Step '+str(counter))
            alldict=np.load(f,allow_pickle=True)
            lons=alldict['arr_2']
            lats=alldict['arr_1']
            singlewave=np.round(alldict['arr_0'],decimals=2)
            if startype=='Temp':
                savefile=savepath+'temp_wave_Teff={}_{:0.2f}'.format(int(Tstar),singlewave)
            elif startype=='Model':
                savefile=savepath+'temp_wave_Model_{:0.2f}'.format(singlewave)
            if os.path.exists(savefile+'.npz'):
                print('Found the previously saved file {}. Now exiting.'.format(savefile))
                prevfile=np.load(savefile+'.npz')
                lats=prevfile['arr_1']
                lons=prevfile['arr_2']
                mapintemp=prevfile['arr_3']

            else:
                fullMapArray=alldict['arr_4']
                bestMapArray=alldict['arr_3']
                lons=alldict['arr_2']
                lats=alldict['arr_1']
                inputmap=np.percentile(fullMapArray,percentiles,axis=0)
                inputmap[1,:,:]=bestMapArray
                mapintemp=np.zeros(np.shape(inputmap))
                inputmap[inputmap<0.]=0.
                wavesround=np.round(waves,decimals=2)
                twaveInd=np.where(wavesround==singlewave)[0][0]
                for medindex in np.arange(np.shape(mapintemp)[0]):
                    wavegrid=np.linspace(waves[twaveInd]-dlam[twaveInd]/2.,waves[twaveInd]+dlam[twaveInd]/2.,1000)
                    throughput=calc_instrument_throughput('NIRISS',wavegrid)
                    if startype=='Temp':
                        mapintemp[medindex]=convert_temp(wavegrid,waves[twaveInd],inputmap[medindex],rprs,throughput,startype=startype,Tstar=Tstar)#fcorr,fpfs,
                    elif startype=='Model':
                        mapintemp[medindex]=convert_temp(wavegrid,waves[twaveInd],inputmap[medindex],rprs,throughput,startype=startype,smodel=smodel)
                np.savez(savefile,alldict['arr_0'],lats,lons,mapintemp)
            
            goodplot=np.where((lons[0,:]>extent[0])&(lons[0,:]<extent[1]))[0]
            tmaxtemp=np.max(mapintemp[:,:,np.min(goodplot):np.max(goodplot)+1])
            tmintemp=np.min(mapintemp[:,:,np.min(goodplot):np.max(goodplot)+1])
            if tmaxtemp>maxtemp:
                maxtemp=tmaxtemp
            if tmintemp<mintemp:
                mintemp=tmintemp

            counter+=1

        if mintemp<1500.:
            mintemp=1500.
        if startype=='Temp':
            flist2=glob.glob(os.path.join(savepath,'temp*Teff={}*.npz'.format(int(Tstar))))
        elif startype=='Model':
            flist2=glob.glob(os.path.join(savepath,'temp*Model*.npz'))
        for f in flist2:
            alldict=np.load(f,allow_pickle=True)
            singlewave=alldict['arr_0']
            mapintemp=alldict['arr_3']

            rc('axes',linewidth=2)
            fig=plt.figure(figsize=(10,6.5))
            ax=plt.axes([0.1,0.35,0.8,0.6],projection=ccrs.Robinson()) 
            plotextent = np.array([np.min(lons[0,goodplot])/np.pi*180,np.max(lons[0,goodplot])/np.pi*180,np.min(lats[:,0])/np.pi*180,np.max(lats[:,0])/np.pi*180])
            vis = calc_contribution_allang(lats[:,goodplot],lons[:,goodplot],mincentlon,maxcentlon)
            plotData = ax.imshow(mapintemp[1][:,goodplot], extent=plotextent,vmin=mintemp,vmax=maxtemp, alpha=vis, transform=ccrs.PlateCarree(),cmap='inferno')
            plt.title('Wavelength={:0.2f}$\mu$m'.format(singlewave),fontsize=15)
            gl=ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True)
            gl.xlabel_style = {'size': 15, 'color': 'k'}
            gl.ylabel_style = {'size': 15, 'color': 'k'}
            
            rc('axes',linewidth=1)
            a = plt.axes([0.2,0.1,0.6,0.15])
            colorbardata=np.zeros((100,100))
            colorbaralpha=np.zeros((100,100))
            for i in np.arange(100):
                colorbardata[i,:]=np.linspace(mintemp,maxtemp,100)
                colorbaralpha[:,i]=np.linspace(0,1,100)
            cbarextent=np.array([mintemp,maxtemp,1.,0.])
            astuff = a.imshow(colorbardata, alpha=colorbaralpha,aspect='auto',extent=cbarextent,cmap='inferno')#, visible=False)
            a.set_xlabel('Temperature [K]',fontsize=15)
            a.set_ylabel('Contribution',fontsize=15)
            a.tick_params(labelsize=15,axis="both",top=True,right=True,width=1,length=4,direction='in')

            if startype=='Temp':
                plt.savefig(savepath+'/retrieved_maps/tempmap_{}_Teff={}_wave_{:0.2f}.pdf'.format(saveName,int(Tstar),singlewave))
            elif startype=='Model':
                plt.savefig(savepath+'/retrieved_maps/tempmap_{}_Model_wave_{:0.2f}.pdf'.format(saveName,singlewave))
            plt.show()

            fig, axArr = plt.subplots(1,3,figsize=(22,5))
            for ind,onePercentile in enumerate(percentiles):
                plotData = axArr[ind].imshow(mapintemp[ind][:,goodplot], extent=plotextent,vmin=mintemp,vmax=maxtemp,cmap='inferno')
                axArr[ind].set_ylabel('Latitude')
                axArr[ind].set_xlabel('Longitude')
                axArr[ind].set_title("{} %".format(onePercentile))
            fig.suptitle('{:.2f}$\mu$m'.format(singlewave),fontsize=20)
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(plotData, cax=cbar_ax)
            cbar.set_label('Temperature [K]')
            if startype=='Temp':
                plt.savefig(savepath+'/retrieved_maps/3_tempmaps_{}_Teff={}_wave_{:0.2f}.pdf'.format(saveName,int(Tstar),singlewave))
            elif startype=='Model':
                plt.savefig(savepath+'/retrieved_maps/3_tempmaps_{}_Model_wave_{:0.2f}.pdf'.format(saveName,singlewave))
            plt.show()

    return mapintemp

def calc_instrument_throughput(instr,waves,order=1):
    #Calculate instrument throughput using pandeia, but here just downloading a fixed file
    #Only works for NIRISS SOSS right now - would need to add other instruments
    # from pandeia.engine.instrument_factory import InstrumentFactory

    if instr == 'NIRISS':
        if order == 1:
            throughputfile = np.loadtxt('./throughputs/niriss_order1.txt')
            throughput = np.interp(waves,throughputfile[:,0],throughputfile[:,1])
    #     conf={"detector": {"nexp": 1,"ngroup": 10,"nint": 1,"readout_pattern": "nisrapid","subarray": "substrip96"},
    # "dynamic_scene": False,"instrument": {"aperture": "soss","disperser": "gr700xd","filter": "clear","instrument": "niriss","mode": "soss"},}

    # instrument_factory = InstrumentFactory(config=conf)
    # if instr == 'NIRISS':
    #     instrument_factory.order = 1
    # throughput = instrument_factory.get_total_eff(waves)

    return throughput

def find_hotspot(outputpath,waves,min_lon,max_lon,step_size,waveInd=3,saveName=None):
    flist=glob.glob(os.path.join(outputpath,'wave*.npz'))
    percentiles = [2.5,16,50,84,97.5]

    if os.path.exists(outputpath+'/hotspot_loc') == False:
        os.makedirs(outputpath+'/hotspot_loc')

    if isinstance(waveInd,int):
        wavesround=np.round(waves[waveInd],decimals=2)
        for f in flist:
            if float(f[-8:-4])==wavesround:
                goodf=f 
        alldict=np.load(goodf,allow_pickle=True)
        bestmap=alldict['arr_3']
        lons=alldict['arr_2']
        lats=alldict['arr_1']
        fullmap=alldict['arr_4']
        singlewave=alldict['arr_0']
        hotlats=np.zeros(np.shape(fullmap)[0])
        hotlons=np.zeros(np.shape(fullmap)[0])
        for i in np.arange(np.shape(fullmap)[0]):
            goodmap=fullmap[i,:,:]
            hottestpoint=np.unravel_index(goodmap.argmax(),goodmap.shape)
            hotlats[i]=lats[hottestpoint[0],hottestpoint[1]]*180./np.pi
            hotlons[i]=lons[hottestpoint[0],hottestpoint[1]]*180./np.pi
        bestpoint=np.unravel_index(bestmap.argmax(),bestmap.shape)
        bestlat=lats[bestpoint[0],bestpoint[1]]*180./np.pi 
        bestlon=lons[bestpoint[0],bestpoint[1]]*180./np.pi

        latsigma=np.percentile(hotlats,percentiles)
        lonsigma=np.percentile(hotlons,percentiles)

        bin_edges=np.arange(min_lon,max_lon,step_size)
        rc('axes',linewidth=2)
        plt.figure()
        plt.hist(hotlats,bins=bin_edges,color='b',label='Latitude',alpha=0.5)
        plt.hist(hotlons,bins=bin_edges,color='r',label='Longitude',alpha=0.5)
        plt.axvline(x=bestlat,color='b',linewidth=3,linestyle='--')
        plt.axvline(x=bestlon,color='r',linewidth=3,linestyle='--')
        plt.tick_params(labelsize=20,axis="both",top=True,right=True,width=2,length=8,direction='in')
        plt.ylabel('Number of Samples',fontsize=20)
        plt.xlabel('Hotspot Location',fontsize=20)
        plt.legend(fontsize=15)
        plt.tight_layout()
        plt.savefig(outputpath+'hotspot_loc/hotspot_loc_{}_waveInd_{:0.2f}.pdf'.format(saveName,singlewave))
        plt.show()

    elif waveInd=='Full':
        latsigma=np.zeros((len(flist),len(percentiles)))
        lonsigma=np.zeros((len(flist),len(percentiles)))
        bin_edges=np.arange(min_lon,max_lon,step_size)
        counter=0
        for f in flist:
            alldict=np.load(f,allow_pickle=True)
            bestmap=alldict['arr_3']
            lons=alldict['arr_2']
            lats=alldict['arr_1']
            fullmap=alldict['arr_4']
            singlewave=alldict['arr_0']
            hotlats=np.zeros(np.shape(fullmap)[0])
            hotlons=np.zeros(np.shape(fullmap)[0])
            for i in np.arange(np.shape(fullmap)[0]):
                goodmap=fullmap[i,:,:]
                hottestpoint=np.unravel_index(goodmap.argmax(),goodmap.shape)
                hotlats[i]=lats[hottestpoint[0],hottestpoint[1]]*180./np.pi
                hotlons[i]=lons[hottestpoint[0],hottestpoint[1]]*180./np.pi
            bestpoint=np.unravel_index(bestmap.argmax(),bestmap.shape)
            bestlat=lats[bestpoint[0],bestpoint[1]]*180./np.pi 
            bestlon=lons[bestpoint[0],bestpoint[1]]*180./np.pi
            # pdb.set_trace()

            latsigma[counter,:]=np.percentile(hotlats,percentiles)
            lonsigma[counter,:]=np.percentile(hotlons,percentiles)
            counter+=1

            rc('axes',linewidth=2)
            plt.figure()
            plt.hist(hotlats,bins=bin_edges,color='b',label='Latitude',alpha=0.5)
            plt.hist(hotlons,bins=bin_edges,color='r',label='Longitude',alpha=0.5)
            plt.axvline(x=bestlat,color='b',linewidth=3,linestyle='--')
            plt.axvline(x=bestlon,color='r',linewidth=3,linestyle='--')
            plt.tick_params(labelsize=20,axis="both",top=True,right=True,width=2,length=8,direction='in')
            plt.ylabel('Number of Samples',fontsize=20)
            plt.xlabel('Hotspot Location',fontsize=20)
            plt.legend(fontsize=15)
            plt.tight_layout()
            plt.savefig(outputpath+'hotspot_loc/hotspot_loc_{}_waveInd_{:0.2f}.pdf'.format(saveName,singlewave))
            plt.show()

    return latsigma,lonsigma


def find_groups(dataDir,waves,extent,numsamp,ngroups=4,sortMethod='avg',letters=False):
    """
    Find the eigenspectra using k means clustering

    Parameters
    ----------
    ngroups: int
        Number of eigenspectra to group results into
    degree: int
        Spherical harmonic degree to draw samples from
    testNum: int
        Test number (ie. lightcurve number 1,2, etc.)
    trySamples: int
        Number of samples to find groups with
        All samples take a long time so this takes a random
        subset of samples from which to draw posteriors
    sortMethod: str
        Method to sort the groups returned by K means clustering
        None, will not sort the output
        'avg' will sort be the average of the spectrum
        'middle' will sort by the flux in the middle of the spectrum
    extent: time covered by the eclipse/phase curve.
        Sets what portion of the map is used for clustering (e.g. full planet or dayside only)
    """

    flist=glob.glob(os.path.join(dataDir,"wave*.npz"))
    getsize=np.load(flist[0],allow_pickle=True)
    lats=getsize['arr_1']
    maparray=np.zeros((numsamp,len(flist),np.shape(lats)[0],np.shape(lats)[1]))

    for f in flist:
        alldict=np.load(f,allow_pickle=True)
        wave=alldict['arr_0']
        lats=alldict['arr_1']
        lons=alldict['arr_2']
        tempmaps=alldict['arr_4']
        insertindex=np.argmin(abs(waves-wave))
        nummaps=np.shape(tempmaps)[0]
        randomIndices = np.random.choice(nummaps,numsamp,replace=False)
        maparray[:,insertindex,:,:]=tempmaps[randomIndices,:,:]

    #at this point we have an array that is num samples x waves x lat x long
    goodplot=np.where((lons[0,:]>extent[0])&(lons[0,:]<extent[1]))[0]
    cormaparray = maparray[:,:,:,goodplot] #throw out non-visible longitudes
    lats=lats[:,goodplot]
    lons=lons[:,goodplot]

    # eigenspectra_draws = []
    kgroup_draws = np.zeros((numsamp,np.shape(lats)[0],np.shape(lats)[1]))
    # uber_eigenlist=[[[[] for i in np.arange(len(waves))] for i in range(ngroups)] for i in range(numsamp)]
    eigenspectra_draws = np.zeros((numsamp,ngroups,len(waves)))
    t0=time.time()
    for drawInd in np.arange(numsamp):
        kgroups = kmeans.kmeans(cormaparray[drawInd], ngroups)
        # pdb.set_trace()
        # eigenspectra_draws.append(eigenspectra)
        # kgroup_draws.append(kgroups)
        kgroup_draws[drawInd,:,:]=kgroups
        # pdb.set_trace()
        for group in range(ngroups):
            ok = np.where(kgroups==group)
            eigenspectra_draws[drawInd,group,:] = np.mean(cormaparray[drawInd,:,ok[0],ok[1]])
        # for groupind in range(ngroups):
        #     uber_eigenlist[drawInd][groupind][:]=eigenlist[groupind][:,:]
            # pdb.set_trace()
        if drawInd % 100 == 0:
            t1=time.time()
            print('Step '+str(drawInd))
            print(t1-t0)
            t0=t1

    sortedkgroups=kmeans.sort_draws(eigenspectra_draws,kgroup_draws)
    #Plot mean grouping and per point histograms
    # np.savez('ExFig2_2groups',sortedkgroups,lats,lons)
    # pdb.set_trace()
    show_group_histos(dataDir,sortedkgroups,lats,lons,numsamp,ngroups,letters=letters)

    # eigenspectra,eigenlist = bin_eigenspectra.bin_eigenspectra(cormaparray[drawInd], kgroups, lats, lons)
    intspec,interr,round_kgroups,fullplanetspec,fullplaneterr = bin_eigenspectra.bin_eigenspectra(cormaparray,sortedkgroups,lats,lons,ngroups)

    show_spectra_of_groups(dataDir,waves,intspec,interr,numsamp,ngroups)
    # pdb.set_trace()
    # pdb.set_trace()
    # if sortMethod is not None:
    #     eigenspectra_draws_final, kgroup_draws_final,uber_eigenlist_final = \
    #     kmeans.sort_draws(eigenspectra_draws,kgroup_draws,uber_eigenlist,\
    #         method=sortMethod)
    # else:
    #     eigenspectra_draws_final, kgroup_draws_final,uber_eigenlist_final = \
    #     eigenspectra_draws, kgroup_draws,uber_eigenlist
    # pdb.set_trace()
    maps_mean = np.average(cormaparray, axis=0)
    maps_error = np.std(cormaparray, axis=0)

    np.savez(dataDir+'eigenspectra_{}_draws_{}_groups'.format(numsamp,ngroups),\
        intspec,interr,round_kgroups,lats,lons,waves,maps_mean,maps_error)
    return intspec,interr,round_kgroups,fullplanetspec,fullplaneterr

def show_group_histos(savedir,kgroup_draws,lats,lons,numsamp,ngroups,
                      xLons=[-1.39,-0.5,0.5,1.39],
                      xLats=[0.,0.,0.,0.],letters=False):
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
    # file=np.load(savedir+'eigenspectra_{}_draws_{}_groups.npz'.format(numsamp,ngroups),allow_pickle=True)
    # kgroup_draws=file['arr_1']

    # file2=np.load(savedir+'kgroups_{}_draws_{}_groups.npz'.format(numsamp,ngroups),allow_pickle=True)
    input_map=np.mean(kgroup_draws,axis=0)#file2['arr_0']
    # lats=file2['arr_1']
    # lons=file2['arr_2']
    print(np.min(input_map),np.max(input_map))
    cmap=cmr.get_sub_cmap('inferno',0.0,0.8)
    londim = input_map.shape[1]
    windowLabels = ['A','B','C','D']
    plotextent = np.array([np.min(lons[0,:])/np.pi*180,np.max(lons[0,:])/np.pi*180,np.min(lats[:,0])/np.pi*180,np.max(lats[:,0])/np.pi*180])
    
    rc('axes',linewidth=2)
    # plt.figure()
    # plotData = plt.imshow(input_map, extent=plotextent,cmap=cmap,vmin=0,vmax=ngroups-1)
    # cbar = plt.colorbar(plotData)
    # cbar.set_label('Mean Group',fontsize=20)
    # cbar.ax.tick_params(labelsize=15,width=2,length=6)
    # if letters:
    #     for ind in np.arange(len(xLons)):
    #         xLon, xLat = xLons[ind], xLats[ind]
    #         iLon, iLat = np.argmin(np.abs(lons[0,:] - xLon)), np.argmin(np.abs(lats[:,0] - xLat))
    #         plt.text(lons[0,iLon]* 180./np.pi,lats[iLat,0]* 180./np.pi,windowLabels[ind],
    #             color='white')
    # plt.ylabel('Latitude',fontsize=20)
    # plt.xlabel('Longitude',fontsize=20)
    # plt.tick_params(labelsize=20,width=2,length=8)
    # plt.savefig(savedir+'meangroup_{}_draws_{}_groups.pdf'.format(numsamp,ngroups))
    # plt.show()

    fig=plt.figure(figsize=(10,6.5))
    ax=plt.axes([0.1,0.3,0.8,0.65],projection=ccrs.Robinson())
    plotData = ax.imshow(input_map, extent=plotextent,cmap=cmap,vmin=0,vmax=ngroups-1,\
        transform=ccrs.PlateCarree())
    gl=ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True)
    gl.xlabel_style = {'size': 15, 'color': 'k'}
    gl.ylabel_style = {'size': 15, 'color': 'k'}

    rc('axes',linewidth=1)
    a = plt.axes([0.2,0.1,0.6,0.1])
    colorbardata=np.zeros((100,100))
    for i in np.arange(100):
        colorbardata[i,:]=np.linspace(0,ngroups-1,100)
    cbarextent=np.array([0,ngroups-1,1.,0.])
    astuff = a.imshow(colorbardata,aspect='auto',extent=cbarextent,cmap=cmap)
    a.yaxis.set_visible(False)
    # a.xaxis.set_visible(False)
    # a=plt.imshow([[np.min(input_map),np.max(input_map)],[np.min(input_map),np.max(input_map)]],cmap=cmap,aspect='auto', visible=False)
    # cbar = plt.colorbar(a)
    # pdb.set_trace()
    a.set_xlabel('Mean Group',fontsize=15)
    # a.set_ylabel('Contribution',fontsize=15)
    a.tick_params(labelsize=15,axis="both",top=True,right=True,width=1,length=4,direction='in')
    # cbar.set_label('Mean Group',fontsize=20)
    # cbar.ax.tick_params(labelsize=15,width=2,length=6)
    if letters:
        for ind in np.arange(len(xLons)):
            xLon, xLat = xLons[ind], xLats[ind]
            iLon, iLat = np.argmin(np.abs(lons[0,:] - xLon)), np.argmin(np.abs(lats[:,0] - xLat))
            ax.text(lons[0,iLon]* 180./np.pi,lats[iLat,0]* 180./np.pi,windowLabels[ind],
                color='white',transform=ccrs.PlateCarree())
    # pdb.set_trace()
    # plt.ylabel('Latitude',fontsize=20)
    # plt.xlabel('Longitude',fontsize=20)
    # plt.tick_params(labelsize=20,width=2,length=8)
    plt.savefig(savedir+'meangroup_{}_draws_{}_groups.pdf'.format(numsamp,ngroups))
    plt.show()
    # pdb.set_trace()


    fig,axs=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True)
    for ind in np.arange(len(xLons)):
        xLon, xLat = xLons[ind], xLats[ind]
        # left, bottom, width, height = [windowLocationsX[ind], windowLocationsY[ind], 0.2, 0.2]
        # ax2 = fig.add_axes([left, bottom, width, height])
        iLon, iLat = np.argmin(np.abs(lons[0,:] - xLon)), np.argmin(np.abs(lats[:,0] - xLat))
        p1=int(ind/2)
        p2=int(ind-p1*2)
        histstats=axs[p1][p2].hist(kgroup_draws[:,iLat,iLon],bins=np.linspace(-0.5,ngroups-0.5,ngroups+1))
        axs[p1][p2].set_title(windowLabels[ind],fontsize=20)
        axs[p1][p2].set_xlim(-0.5,np.max(kgroup_draws) + 0.5)
        print('Percentage in Maximum Group, point '+str(ind)+': '+str(float(np.max(histstats[0]))/float(numsamp)*100.))
        # pdb.set_trace()
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

def show_spectra_of_groups(savedir,waves,spec,err,numsamp,ngroups):
    """
    Calculate the mean and standard deviation of the spectra
    as well as the kgroups map
    Plot the mean and standard deviations of the spectra
    """
    #eigenspectra = np.mean(eigenspectra_draws, axis=0)
    #eigenerrs = np.std(eigenspectra_draws, axis=0)
    # file=np.load(savedir+'eigenspectra_{}_draws_{}_groups.npz'.format(numsamp,ngroups),allow_pickle=True)
    # eigenspectra_draws=file['arr_0']
    # kgroup_draws=file['arr_1']
    # uber_eigenlist=file['arr_2']
    # maps=file['arr_3']
    # lats=file['arr_4']
    # lons=file['arr_5']
    # # pdb.set_trace()

    # kgroups = np.mean(kgroup_draws, axis=0)
    # perpointspec = np.mean(maps,axis=0)
    # perpointerr = np.std(maps,axis=0)
    # nearestint= np.round(kgroups,decimals=0)
    # eigenspec = np.zeros((ngroups,len(waves)))
    # eigenerr = np.zeros((ngroups,len(waves)))
    # for g in range(ngroups):
    #     ingroup = np.where(nearestint==g)
    #     eigenspec[g,:] = np.sum(perpointspec[:,ingroup[0],ingroup[1]] * \
    #         np.cos(lats[ingroup[0],ingroup[1]]),axis=1)/np.sum(np.cos(lats[ingroup[0],ingroup[1]]))
    #     eigenerr[g,:] = np.sum(perpointerr[:,ingroup[0],ingroup[1]] * \
    #         np.cos(lats[ingroup[0],ingroup[1]]),axis=1)/np.sum(np.cos(lats[ingroup[0],ingroup[1]]))
    # pdb.set_trace()

    # #integrate over full sphere
    # centlon=0.
    # centlat=0.
    # dellat=np.diff(lats[:,0])[0]
    # dellon=np.diff(lons[0,:])[0]
    # centlat_sin = np.sin(centlat)
    # centlat_cos = np.cos(centlat)
    # vis = centlat_sin * np.sin(lats) + centlat_cos * np.cos(lats) * \
    #         np.cos(lons - centlon)
    # vis[vis <= 0.] = 0.
    # integratedspec = np.zeros((ngroups,len(waves)))
    # integratederr = np.zeros((ngroups,len(waves)))
    # for g in range(ngroups):
    #     integratedspec[g,:] = eigenspec[g,:]*np.sum(vis*np.cos(lats)) * dellat * dellon
    #     integratederr[g,:] = eigenerr[g,:]*np.sum(vis*np.cos(lats)) * dellat * dellon

    # pdb.set_trace()
    # allsamples=[[[] for i in range(np.shape(waves)[0])] for i in range(np.shape(uber_eigenlist)[1])]
    # for x in range(np.shape(uber_eigenlist)[0]):
    #     for y in range(np.shape(uber_eigenlist)[1]):
    #         for z in range(np.shape(uber_eigenlist)[2]):
    #             allsamples[y][z]=np.concatenate((allsamples[y][z],uber_eigenlist[x][y][z]))

    # eigenspectra=np.zeros((np.shape(allsamples)[0],np.shape(allsamples)[1]))
    # eigenerrs=np.zeros((np.shape(allsamples)[0],np.shape(allsamples)[1]))
    # for x in range(np.shape(allsamples)[0]):
    #     for y in range(np.shape(allsamples)[1]):
    #         eigenspectra[x,y]=np.mean(allsamples[x][y])
    #         eigenerrs[x,y]=np.std(allsamples[x][y])
    # eigenspectra=np.mean(eigenspectra_draws,axis=0)
    # eigenerrs=np.std(eigenspectra_draws,axis=0)
    # print(integratedspec*10**6.,integratederr*10**6.)
    #Hm....ok I need to think more carefully about this.
    #print(np.shape(kgroups))
    #waves=np.array([2.41,2.59,2.77,2.95,3.13,3.31,3.49,3.67,3.85,4.03])
    #print(kgroups)
    #print(np.min(kgroups),np.max(kgroups))
    #print(np.around(np.min(kgroups)),np.around(np.max(kgroups)))
    #outfile=open('mystery4_eigenspectra5.txt','w')
    counter=0
    colors=pl.cm.inferno(np.linspace(0,0.8,ngroups))
    rc('axes',linewidth=2)
    fig, ax = plt.subplots()
    for i in range(ngroups):
        ax.errorbar(waves, spec[i,:]*10.**6., err[i,:]*10.**6.,label=('Group '+np.str(counter)),linewidth=2,marker='.',markersize=10,color=colors[counter])
        counter+=1
        #for i in np.arange(np.shape(waves)[0]):
        #    print(waves[i],spec[i],err[i],file=outfile)
    ax.set_xlabel('Wavelength ($\mu$m)',fontsize=20)
    ax.set_ylabel('F$_p$/F$_*$ [ppm]',fontsize=20)
    ax.tick_params(labelsize=20,axis="both",top=True,right=True,width=2,length=8,direction='in')
    #ax.set_title('Eigenspectra')
    ax.legend(fontsize=15)

    #outfile.close()
    # Ngroup = eigenspectra_draws.shape[1]
    fig.savefig(savedir+'spectra_{}_draws_{}_groups.pdf'.format(numsamp,ngroups),bbox_inches='tight')
    plt.show()
    # np.savez(savedir+'kgroups_{}_draws_{}_groups'.format(numsamp,ngroups),kgroups,\
    #     lats,lons,maps)
    return #kgroups,lats,lons,maps

def group_spectra_in_temp(datadir,numsamp,ngroups,dlam,rprs,startype='Temp',Tstar=6000.,smodel=None):
    file=datadir+'eigenspectra_{}_draws_{}_groups.npz'.format(int(numsamp),int(ngroups))
    assert (startype=='Temp') | (startype=='Model'), "Only startype='Temp' and startype='Model' are supported options."

    try:
        spec=np.load(file)
    except:
        print("Error: could not find file "+str(file))

    intspec=spec['arr_0']
    interr=spec['arr_1']
    waves=spec['arr_5']
    # np.savez(dataDir+'eigenspectra_{}_draws_{}_groups'.format(numsamp,ngroups),\
    #     intspec,interr,round_kgroups,lats,lons,waves,maps_mean,maps_error)
    # pdb.set_trace()
    tempspec=np.zeros(np.shape(intspec))
    temperr=np.zeros((np.shape(interr)[0],np.shape(interr)[1],2))
    for i in np.arange(np.shape(intspec)[1]):
        wavegrid=np.linspace(waves[i]-dlam[i]/2.,waves[i]+dlam[i]/2.,1000)
        throughput=calc_instrument_throughput('NIRISS',wavegrid)
        for j in np.arange(np.shape(intspec)[0]):
            if startype=='Temp':
                tempspec[j,i]=convert_temp_spectra(wavegrid,waves[i],intspec[j,i],rprs,throughput,startype=startype,Tstar=Tstar)#fpfs,fcorr,
                temperr[j,i,1]=convert_temp_spectra(wavegrid,waves[i],intspec[j,i]+interr[j,i],rprs,throughput,startype=startype,Tstar=Tstar)-tempspec[j,i]
                temperr[j,i,0]=tempspec[j,i]-convert_temp_spectra(wavegrid,waves[i],intspec[j,i]-interr[j,i],rprs,throughput,startype=startype,Tstar=Tstar)
            elif startype=='Model':
                tempspec[j,i]=convert_temp_spectra(wavegrid,waves[i],intspec[j,i],rprs,throughput,startype=startype,smodel=smodel)#fpfs,fcorr,
                temperr[j,i,1]=convert_temp_spectra(wavegrid,waves[i],intspec[j,i]+interr[j,i],rprs,throughput,startype=startype,smodel=smodel)-tempspec[j,i]
                temperr[j,i,0]=tempspec[j,i]-convert_temp_spectra(wavegrid,waves[i],intspec[j,i]-interr[j,i],rprs,throughput,startype=startype,smodel=smodel)

    rc('axes',linewidth=2)
    counter=0
    colors=pl.cm.inferno(np.linspace(0,0.8,ngroups))
    plt.figure()
    plt.tick_params(labelsize=20,axis="both",top=True,right=True,width=2,length=8,direction='in')
    for i in range(ngroups):
        plt.errorbar(waves, tempspec[i,:], temperr[i,:,:].T,label=('Group '+np.str(counter)),linewidth=2,marker='.',markersize=10,color=colors[counter])
        counter+=1
    plt.xlabel('Wavelength ($\mu$m)',fontsize=20)
    plt.ylabel('Temperature (K)', fontsize=20)
    # plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(datadir+'temp_spectra_{}_draws_{}_groups.pdf'.format(numsamp,ngroups))
    plt.show()

    return tempspec,temperr

def convert_temp_spectra(wavegrid,centwave,inputmap,rprs,throughput,startype='Temp',Tstar=6000.,smodel=None): #fpfs,fcorr,#wavegrid in microns
    from scipy.optimize import leastsq
    c=2.998*10**8.
    h=6.626*10**-34.
    kb=1.381*10**-23.
    wavesinm=wavegrid*10**-6.
    if startype=='Temp':
        starbb=2*h*c**2./wavesinm**5./(np.exp(h*c/(wavesinm*kb*Tstar))-1)
        starint=specint(wavesinm,starbb,throughput)
    elif startype=='Model':
        #assume that model is in correct units (wavelengths in m and flux in W/m^3/sr)
        starbb = np.interp(wavesinm,smodel[:,0],smodel[:,1])
        starint=specint(wavesinm,starbb,throughput)
    centwave*=10**-6.
    # ptemp=h*c/(centwave*kb) #FINDME: THIS IS THE LAZY WAY
    # mapintemp1=np.zeros(np.shape(inputmap))
    # for i in np.arange(np.shape(inputmap)[0]):
    #     for j in np.arange(np.shape(inputmap)[1]):
    #         mapintemp1[i,j]=ptemp/np.log(1+rprs**2.*(2*h*c**2.)\
    #             /(centwave**5.*np.pi*inputmap[i,j]*fcorr*starint*fpfs))
    theflux=starint*inputmap/rprs**2.#fpfs*fcorr
    temp0=np.array([1000.])
    tfit=leastsq(planetbbod,temp0,args=(theflux,wavesinm,throughput))
    mapintemp=tfit[0]
    return mapintemp

def do_hue_maps(savedir,numsamp,ngroups,hueType='group'):
    #full_extent = np.array([np.min(lons),np.max(lons),np.min(lats),np.max(lats)])/np.pi*180 #for full map
    #full_extent = np.array([-90.,90.,-90.,90.]) #for dayside only
    file=np.load(savedir+'eigenspectra_{}_draws_{}_groups.npz'.format(numsamp,ngroups),allow_pickle=True)
    # eigenspectra_draws=file['arr_0']
    # kgroup_draws=file['arr_1']
    # uber_eigenlist=file['arr_2']
    # maps=file['arr_3']
    lats=file['arr_3']
    lons=file['arr_4']
    waves=file['arr_5']
    # kgroups = np.mean(kgroup_draws, axis=0)
    kround = file['arr_2']
    # full_extent = np.array([-extent/2.*360.,extent/2.*360.,-90.,90.])
    # londim, latdim = np.shape(maps)[1:]
    # maps = #file['arr_6']
    maps_mean = file['arr_6']#np.average(maps, axis=0)
    maps_error = file['arr_7']#np.std(maps, axis=0)

    # cmap = cmr.get_sub_cmap('inferno',0.0,0.8)#cc.cm['isolum']
    # cmap_grey = cc.cm['linear_grey_10_95_c0']
    # cmap_grey_r = cc.cm['linear_grey_10_95_c0_r']
    # norm = Normalize(vmin=np.min(maps_mean), vmax=np.max(maps_mean))
    # londim=100

    # kround=np.around(kgroups)
    # minlon=np.around(extent/2.*londim)

    # contlons=lons[:,int(londim/2.-minlon):int(londim/2.+minlon)] #?
    # contlats=lats[:,int(londim/2.-minlon):int(londim/2.+minlon)] #?
    # pdb.set_trace()
    plotextent = np.array([np.min(lons[0,:])/np.pi*180,np.max(lons[0,:])/np.pi*180,np.min(lats[:,0])/np.pi*180,np.max(lats[:,0])/np.pi*180])
    cmap=cmr.get_sub_cmap('inferno',0.0,0.8)
    colors=pl.cm.inferno(np.linspace(0,0.8,ngroups))

    mincentlon=np.min(lons[0,:])+np.pi/2. #minimum subobserver point, in radians
    maxcentlon=np.max(lons[0,:])-np.pi/2.
######## STUFF ON HOW TO PLOT FROM ABOVE
# rc('axes',linewidth=2)
# fig=plt.figure(figsize=(10,6.5))
# ax=plt.axes([0.1,0.35,0.8,0.6],projection=ccrs.Robinson())
# # plt.figure()
# map_day = mapLowMedHigh[1][:,goodplot]
# plotextent = np.array([np.min(lons[0,goodplot])/np.pi*180,np.max(lons[0,goodplot])/np.pi*180,np.min(lats[:,0])/np.pi*180,np.max(lats[:,0])/np.pi*180])
# # plotData = plt.imshow(map_day, extent=plotextent,vmin=minflux,vmax=maxflux)
# vis = calc_contribution(lats[:,goodplot],lons[:,goodplot])
# plotData = ax.imshow(map_day, extent=plotextent,vmin=minflux,vmax=maxflux, alpha=vis, transform=ccrs.PlateCarree())
# # cbar = plt.colorbar(plotData)
# plt.title('Wavelength={:0.2f} $\mu$m'.format(waves[waveInd]),fontsize=15)
# gl=ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True)
# gl.xlabel_style = {'size': 15, 'color': 'k'}
# gl.ylabel_style = {'size': 15, 'color': 'k'}
# # plt.ylabel('Latitude',fontsize=20)
# # plt.xlabel('Longitude',fontsize=20)
# rc('axes',linewidth=1)
# a = plt.axes([0.2,0.1,0.6,0.15])
# # a.yaxis.set_visible(False)
# # a.xaxis.set_visible(False)
# colorbardata=np.zeros((100,100))
# colorbaralpha=np.zeros((100,100))
# for i in np.arange(100):
#     colorbardata[i,:]=np.linspace(minflux*10**6.,maxflux*10**6.,100)
#     colorbaralpha[:,i]=np.linspace(0,1,100)
# cbarextent=np.array([minflux*10**6.,maxflux*10**6.,1.,0.])
# astuff = a.imshow(colorbardata, alpha=colorbaralpha,aspect='auto',extent=cbarextent)#, visible=False)
# a.set_xlabel('Fp/Fs [ppm]',fontsize=15)
# a.set_ylabel('Contribution',fontsize=15)
# a.tick_params(labelsize=15,axis="both",top=True,right=True,width=1,length=4,direction='in')
# # cbar=plt.colorbar(a, fraction=3.0,orientation='horizontal')
    if hueType == 'group':
        # for index in np.arange(len(waves)):
        rc('axes',linewidth=2)
        fig=plt.figure(figsize=(10,6.5))
        ax=plt.axes([0.1,0.35,0.8,0.6],projection=ccrs.Robinson())
        vis = calc_contribution_allang(lats,lons,mincentlon,maxcentlon)
        # error_range = 1.0-0.5*(maps_error[index]-np.min(maps_error[index]))/(\
        #     np.max(maps_error[index])-np.min(maps_error[index]))#calculating scale ranging from 0 to 1 for relative error
        # pdb.set_trace()
        # vis[error_range<vis]=error_range[error_range<vis]
        # pdb.set_trace()
        plotData = ax.imshow(kround, extent=plotextent,vmin=np.min(kround),vmax=np.max(kround), \
            alpha=vis, transform=ccrs.PlateCarree(),cmap=cmap)
        # plt.title('Wavelength={:0.2f} $\mu$m'.format(waves[index]),fontsize=15)
        gl=ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True)
        gl.xlabel_style = {'size': 15, 'color': 'k'}
        gl.ylabel_style = {'size': 15, 'color': 'k'}
        rc('axes',linewidth=1)
        a = plt.axes([0.2,0.1,0.6,0.15])
        colorbardata=np.zeros((100,ngroups,4))
        alphas=np.linspace(0,1,100)
        for i in np.arange(100):
            colorbardata[i,:,:]=colors
            colorbardata[i,:,3]=alphas[i]
        # for i in np.arange(int(np.max(kround)+1)):
        #     colorbardata[:,i,:]=np.linspace(0,1,100)
        cbarextent=np.array([np.min(kround),np.max(kround),1.,0.])
        astuff = a.imshow(colorbardata, aspect='auto',extent=cbarextent,interpolation='none')
        a.set_xticks(ticks=np.arange(ngroups).tolist())
        groups=np.arange(ngroups)
        strlist=[]
        for i in groups:
            strlist.append(str(i))
        a.set_xticklabels(strlist)
        a.set_xlabel('Group',fontsize=15)
        a.set_ylabel('Contribution',fontsize=15)
        a.tick_params(labelsize=15,axis="both",top=True,right=True,width=1,length=4,direction='in')
        plt.savefig(savedir+'HueGroup_{}_groups.pdf'.format(ngroups),bbox_inches='tight')
        plt.show()

    # from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
    #                             LatitudeLocator)
    # import matplotlib.ticker as mticker
    # if hueType == 'group':
    #     for index in np.arange(len(waves)):
    #         fig=plt.figure(figsize=(10,6.5))
    #         ax=plt.axes(projection=ccrs.Robinson())
    #         # plt.title('Eigengroups', fontsize=20)
    #         #colormap2D is throwing an error somewhere in here.
    #         # ax.set_extent(plotextent,crs=ccrs.PlateCarree())
    #         group_map = generate_map2d(hue_quantity=kround,
    #                                    lightness_quantity=maps_mean[index],
    #                                    hue_cmap=cmap,
    #                                    scale_min=10,
    #                                    scale_max=90)
    #         ax.imshow(group_map, extent=plotextent, interpolation='gaussian',transform=ccrs.PlateCarree())
    #         CS = plt.contour(lons/np.pi*180, lats/np.pi*180, kround,
    #                        levels=np.arange(ngroups), colors='k', linestyles='dashed',transform=ccrs.PlateCarree())

    #         plt.clabel(CS, inline=1, fmt='%1.0f', fontsize=15)

    #         plt.xlabel(r'Longitude ($^\circ$)', fontsize=20)
    #         plt.ylabel(r'Latitude ($^\circ$)', fontsize=20)
    #         plt.title('Wavelength={:0.2f} $\mu$m'.format(waves[index]),fontsize=15)
    #         gl=ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True)
    #         gl.xlabel_style = {'size': 15, 'color': 'k'}
    #         gl.ylabel_style = {'size': 15, 'color': 'k'}
    #         # plt.setp(plt.axes().get_xticklabels(), fontsize=20)
    #         # plt.setp(plt.axes().get_yticklabels(), fontsize=20)

    #         cmap_group = cmap
    #         cNorm_group  = Normalize(vmin=0, vmax=ngroups-1)
    #         scalarMap_group = cm.ScalarMappable(norm=cNorm_group, cmap=cmap_group)

    #         cmap_flux = cmap_grey
    #         cNorm_flux  = Normalize(vmin=0, vmax=np.nanmax(maps_mean))
    #         scalarMap_flux = cm.ScalarMappable(norm=cNorm_flux, cmap=cmap_flux)
    #         pdb.set_trace()
    #         bounds = np.linspace(-0.5, ngroups-0.5, ngroups+1)
    #         norm_group = BoundaryNorm(bounds, cmap_group.N)

    #         # divider = make_axes_locatable(plt.axes())
    #         # ax2 = divider.append_axes("bottom", size="7.5%", pad=1)
    #         # cb = colorbar.ColorbarBase(ax2, cmap=cmap_group, norm=norm_group, spacing="proportional", orientation='horizontal', ticks=np.arange(0, ngroups, 1), boundaries=bounds)
    #         # cb.ax.xaxis.set_minor_formatter(FormatStrFormatter('%.1g'))
    #         # # cb.ax.tick_params(axis='x', labelsize=13)
    #         # cb.ax.tick_params(labelsize=15,axis="both",top=True,right=True,width=2,length=8,direction='in')
    #         # cb.ax.set_title('Group', y=1.35, fontsize=15)

    #         a = plt.axes([0.16,0.16,0.7,0.01], frameon=False)
    #         a.yaxis.set_visible(False)
    #         a.xaxis.set_visible(False)
    #         a = plt.imshow([[0,1],[0,1]], cmap=plt.cm.inferno, aspect='auto', visible=False)
    #         cbar=plt.colorbar(a, fraction=3.0)
    #         cbar.ax.tick_params(labelsize=15,width=2,length=6)
    #         cbar.set_label('Test',fontsize=15)

    #         pdb.set_trace()
    #         ax3 = divider.append_axes("bottom", size="7.5%", pad=0.75)
    #         cb2 = colorbar.ColorbarBase(ax3, cmap=cmap_flux, norm=cNorm_flux, orientation='horizontal')
    #         cb2.ax.tick_params(axis='x', labelsize=15)
    #         cb2.ax.set_title('Flux', y=1.35, fontsize=15)

    #         plt.savefig(savedir+'HUEgroup_LUMflux_{}_groups_wave_{:0.2f}.pdf'.format(ngroups,waves[index]),bbox_inches='tight')
    #         plt.show()
    #         pdb.set_trace() #FINDME: The projection thing works on its own, but not with the extra colorbars at the bottom.
        #for filetype in ['png', 'pdf']:
        #    p.savefig('HUEgroup_LUMflux_quadrant_deg6_group4.{}'.format(filetype), dpi=300, bbox_inches='tight')
    elif hueType == 'flux':
        for index in np.arange(len(waves)):
            maxval=np.max(maps_mean[index])*10**6.
            minval=np.min(maps_mean[index])*10**6.
            rc('axes',linewidth=2)
            fig=plt.figure(figsize=(10,6.5))
            ax=plt.axes([0.1,0.35,0.8,0.6],projection=ccrs.Robinson())
            vis = calc_contribution(lats,lons)
            # error_range = 1.0-0.5*(maps_error[index]-np.min(maps_error[index]))/(\
            #     np.max(maps_error[index])-np.min(maps_error[index]))#calculating scale ranging from 0 to 1 for relative error
            # pdb.set_trace()
            # vis[error_range<vis]=error_range[error_range<vis]
            # pdb.set_trace()
            plotData = ax.imshow(maps_mean[index]*10**6., extent=plotextent,vmin=minval,vmax=maxval, \
                alpha=vis, transform=ccrs.PlateCarree(),cmap=cmap)
            plt.title('Wavelength={:0.2f} $\mu$m'.format(waves[index]),fontsize=15)
            gl=ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True)
            gl.xlabel_style = {'size': 15, 'color': 'k'}
            gl.ylabel_style = {'size': 15, 'color': 'k'}
            rc('axes',linewidth=1)
            a = plt.axes([0.2,0.1,0.6,0.15])
            colorbardata=np.zeros((100,100))
            colorbaralpha=np.zeros((100,100))
            for i in np.arange(100):
                colorbardata[i,:]=np.linspace(minval,maxval,100)
                colorbaralpha[:,i]=np.linspace(0,1,100)
            # pdb.set_trace()
            cbarextent=np.array([minval,maxval,1.,0.])
            astuff = a.imshow(colorbardata, alpha=colorbaralpha,aspect='auto',extent=cbarextent,cmap=cmap)
            # for i in np.arange(int(np.max(kround)+1)):
            #     colorbardata[:,i,:]=np.linspace(0,1,100)
            #a.set_xticks(ticks=np.arange(ngroups).tolist())
            #groups=np.arange(ngroups)
            a.set_xlabel('Fp/Fs [ppm]',fontsize=15)
            a.set_ylabel('Contribution',fontsize=15)
            a.tick_params(labelsize=15,axis="both",top=True,right=True,width=1,length=4,direction='in')
            plt.savefig(savedir+'HueFlux_{}_groups_wave_{:0.2f}.pdf'.format(ngroups,waves[index]),bbox_inches='tight')
            plt.show()
        # for index in np.arange(len(waves)):
        #     plt.figure(figsize=(10,6.5))
        #     plt.title('Flux', fontsize=20)

        #     group_map = generate_map2d(hue_quantity=(maps_mean[index]-np.min(maps_mean[index]))/np.ptp(maps_mean[index]),
        #                                lightness_quantity=1-((maps_error[index]*100.-np.min(maps_error[index]*100.))/np.ptp(maps_error[index]*100.)),
        #                                hue_cmap='inferno',
        #                                scale_min=10,
        #                                scale_max=90)
        #     plt.imshow(group_map, extent=plotextent, interpolation='gaussian')
        #     CS = plt.contour(lons/np.pi*180, lats/np.pi*180, kround,
        #                    levels=np.arange(ngroups), colors='k', linestyles='dashed')

        #     plt.clabel(CS, inline=1, fmt='%1.0f', fontsize=15)

        #     plt.xlabel(r'Longitude ($^\circ$)', fontsize=20)
        #     plt.ylabel(r'Latitude ($^\circ$)', fontsize=20)
        #     plt.title('Wavelength={:0.2f} $\mu$m'.format(waves[index]),fontsize=15)
        #     plt.setp(plt.axes().get_xticklabels(), fontsize=20)
        #     plt.setp(plt.axes().get_yticklabels(), fontsize=20)

        #     cmap_flux = cmap
        #     cNorm_flux = Normalize(vmin=0, vmax=np.nanmax(maps_mean))
        #     scalarMap_flux = cm.ScalarMappable(norm=cNorm_flux, cmap=cmap_flux)

        #     cmap_stdev = cmap_grey_r
        #     cNorm_stdev  = Normalize(vmin=0, vmax=np.nanmax(maps_error*100.))
        #     scalarMap_stdev = cm.ScalarMappable(norm=cNorm_stdev, cmap=cmap_stdev)

        #     divider = make_axes_locatable(plt.axes())
        #     ax2 = divider.append_axes("bottom", size="7.5%", pad=1)
        #     cb = colorbar.ColorbarBase(ax2, cmap=cmap_flux, norm=cNorm_flux, orientation='horizontal')
        #     cb.ax.tick_params(axis='x', labelsize=15)
        #     cb.ax.set_title('Flux', y=1.35, fontsize=15)

        #     ax3 = divider.append_axes("bottom", size="7.5%", pad=0.75)
        #     cb2 = colorbar.ColorbarBase(ax3, cmap=cmap_stdev, norm=cNorm_stdev, orientation='horizontal')
        #     cb2.ax.tick_params(axis='x', labelsize=15)
        #     cb2.ax.set_title('Uncertainty [%]', y=1.35, fontsize=15)

        #     plt.savefig(savedir+'HUEflux_LUMstdev_{}_groups_wave_{:0.2f}.pdf'.format(ngroups,waves[index]),bbox_inches='tight')
        #     plt.show()
        #for filetype in ['png', 'pdf']:
        #    p.savefig('HUEflux_LUMstdev_quadrant_deg6_group4.{}'.format(filetype), dpi=300, bbox_inches='tight')

    else:
        raise Exception("Unrecognized hueType {}".format(hueType))
