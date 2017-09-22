from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib
import glob as glob
from cmocean import cm as cm
import numpy as np


def plot_gcm(vals, land_mask, X_obs=None, levels=None):
    lats = np.arange(-89.375,89.375,1.25)
    longs = np.arange(-180,178.75+0.1, 1.25)
    longgrid, latgrid = np.meshgrid(longs, lats)
    m = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,\
            llcrnrlon=-180,urcrnrlon=180,resolution='c')
    m.drawcoastlines()
    m.drawparallels(np.arange(-90.,91.,30.))
    m.drawmeridians(np.arange(-180.,181.,60.))
    m.drawmapboundary(fill_color='white')

    plt.xlabel('lon')
    plt.ylabel('lat')
    levels = np.arange(-10,10,1)
    yplot = np.zeros(longgrid.size)
    yplot[land_mask-1] = vals

    m.contourf(longgrid,latgrid,yplot.reshape(lats.size,longs.size),15,levels=levels,
    cm = cm.delta)
    m.fillcontinents('black')
    m.colorbar()
    if X_obs is not None:
        mp2.scatter(X_obs[:,0], X_obs[:,1])
    return(m)

def plot_map(longgrid=None, latgrid=None, vals=None, X_obs=None, levels=None):
    mp2 = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,\
        llcrnrlon=-180,urcrnrlon=180,resolution='c')
    mp2.drawcoastlines()
    mp2.drawparallels(np.arange(-90.,91.,30.))
    mp2.drawmeridians(np.arange(-180.,181.,60.))
    mp2.drawmapboundary(fill_color='white')


    plt.xlabel('lon')
    plt.ylabel('lat')
    if levels==None:
        levels = np.arange(-10,12,1)
    if vals is not None:
        mp2.contourf(longgrid,latgrid,vals.reshape(latgrid.shape),15,levels=levels,
    cm = cm.delta)
        mp2.colorbar()

    mp2.fillcontinents('black')

    if X_obs is not None:
        mp2.scatter(X_obs[:,0], X_obs[:,1])
    return(mp2)


def ThinGrid(gcm_output, land_mask, thinby=2, plot=False):
    """
    We don't want to predict at all locations.
    There are 41184 locations in the GCM grid -
    - too many to want to produce a full covariance matrix for.
    Of these, 27186 are ocean, the others are land, but that is still too many.

    As a simple fix, let's just subset by taking every nth value.
    We will also ignore the land and not predict there.

    land_mask should give the location of all the ocean grid cells.


    This approach reduces the number of grid points by approx 1-1/thinby**2

    """
    # create the GCM grid
    lats_gcm = np.arange(-89.375,89.375,1.25) # Is this right?
    longs_gcm = np.arange(-180,178.75+0.1, 1.25)
    longgrid_gcm, latgrid_gcm = np.meshgrid(longs_gcm, lats_gcm)
    yplot = np.zeros(longgrid_gcm.size)-10000.
    yplot[land_mask-1] = gcm_output # IS THIS RIGHT?
    gcm_grid = yplot.reshape(lats_gcm.size,longs_gcm.size)

    keep_lats= np.arange(0,lats_gcm.size,thinby)
    keep_longs= np.arange(0,longs_gcm.size,thinby)

    longgrid_pred = longgrid_gcm[keep_lats,:][:,keep_longs]
    latgrid_pred = latgrid_gcm[keep_lats,:][:,keep_longs]
    gcm_grid_pred = gcm_grid[keep_lats,:][:,keep_longs]

    # create an array of Falses, change the ocean values to true, then thin
    land_mask_TF = np.zeros(longgrid_gcm.shape, dtype=bool)
    tmp = land_mask_TF.flatten()
    tmp[land_mask-1]=True
    land_mask_TF = tmp.reshape(longgrid_gcm.shape)
    land_mask_TF_pred = land_mask_TF[keep_lats,:][:,keep_longs]

    if plot:
        mp2 = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,\
        llcrnrlon=-180,urcrnrlon=180,resolution='c')
        mp2.drawcoastlines()
        mp2.drawparallels(np.arange(-90.,91.,30.))
        mp2.drawmeridians(np.arange(-180.,181.,60.))
        mp2.drawmapboundary(fill_color='white')
        #mp2.scatter(longgrid_pred.flatten(), latgrid_pred.flatten())
        mp2.scatter(longgrid_pred.flatten()[land_mask_TF_pred.flatten()], latgrid_pred.flatten()[land_mask_TF_pred.flatten()])
        plt.show()

    # create the X locations for the prediction grid - thinned and with land removed
    X_pred =np.column_stack((longgrid_pred.flatten()[land_mask_TF_pred.flatten()], latgrid_pred.flatten()[land_mask_TF_pred.flatten()]))

    # return the thinned GCM output.
    gcm_grid_pred_S = gcm_grid_pred.flatten()[land_mask_TF_pred.flatten()]
    if gcm_grid_pred_S.min()<-1000:
        print('Error we have not remvoved all the land successfully')
    return X_pred, gcm_grid_pred_S[:,None]
