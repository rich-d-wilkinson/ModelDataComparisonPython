import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib
import os
import glob as glob
from cmocean import cm as cm

####################
# Load a GCM

#from R: HadCM3grid =  c(-180,178.75,1.25,-89.375,89.375,1.25)

lats = np.arange(-89.375,89.375,1.25)
longs = np.arange(-180,178.75+0.1, 1.25)
longgrid, latgrid = np.meshgrid(longs, lats)


GCM_dir = os.getenv("HOME")+'/Documents/Code/ModelDataComparison/DMC/Scripts/Model_data/CO2_anom/'
gcm_SSTs = glob.glob(GCM_dir+'t*.txt')

gcm_mask = np.genfromtxt(GCM_dir+'mask.txt', dtype='int')
gcm_mask_TF = np.zeros(longgrid.shape, dtype=bool)
tmp = gcm_mask_TF.flatten()
tmp[gcm_mask-1]=True

gcm_mask_TF = tmp.reshape(longgrid.shape)

mp2 = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,\
        llcrnrlon=-180,urcrnrlon=180,resolution='c')
mp2.drawcoastlines()
mp2.drawparallels(np.arange(-90.,91.,30.))
mp2.drawmeridians(np.arange(-180.,181.,60.))
mp2.drawmapboundary(fill_color='white')

mp2.scatter(longgrid.flatten()[gcm_mask_TF.flatten()], latgrid.flatten()[gcm_mask_TF.flatten()]
, s=0.1)

for file_name in gcm_SSTs:
    file_nm = file_name.split(GCM_dir)[-1]
    print(file_nm)

    gcm1 = np.genfromtxt(file_name)
    print('Max + min')
    print(gcm1.max(), gcm1.min())

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
    yplot[gcm_mask-1] = gcm1

    m.contourf(longgrid,latgrid,yplot.reshape(lats.size,longs.size),15,levels=levels,
    cm = cm.delta)
    m.fillcontinents('black')
    m.colorbar()
    #plt.show()
    plt.savefig('figs/'+file_nm[:-4]+'.png')
