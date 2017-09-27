import GPy
import sys
import os
sys.path.append(os.getenv("HOME") + "/Documents/Code/Emulation/GPyDifferentMetrics/")
from HaversineDist import Exponentialhaversine
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib
import glob as glob
from cmocean import cm as cm
from Utilities import *



GCM_dir = os.getenv("HOME")+'/Documents/Code/ModelDataComparison/DMC/Scripts/Model_data/CO2_anom/'
gcm_SSTs = glob.glob(GCM_dir+'t*.txt')

gcm_mask = np.genfromtxt(GCM_dir+'mask.txt', dtype='int')

obs_dir = os.getenv("HOME")+'/Documents/Code/ModelDataComparison/DMC/Scripts/Observation_data/P3+_SST_anom/'
file = 'lambda_10.txt'
observations = np.genfromtxt(obs_dir+file, skip_header=1)

X_obs = observations[:,0:2]
y_obs = observations[:,2].reshape(-1,1)
var_ratios = observations[:,3][:,None]**2

# Plot the data locations
map = plot_map(X_obs=X_obs)
#plt.show()


######################################
# Start by just fitting a GP, not a circle, not with heteroscedastic noise.
k = GPy.kern.Exponential(2, ARD=True)
m = GPy.models.GPRegression(X_obs, y_obs,k)
m.optimize_restarts(10)
print(m)
print(m.rbf.lengthscale)
#

########################################
# Create plotting grid on 5 deg grid
latsplot = np.arange(-85.0,90.0, 5.)
longsplot = np.arange(-180.0,180.0, 5.)
longgridplot, latgridplot = np.meshgrid(longsplot, latsplot)
X_plot=np.column_stack((longgridplot.flatten(), latgridplot.flatten()))

######
# Predict and plot

mu,V = m.predict(X_plot)

plt.figure(1)
map=plot_map(longgridplot, latgridplot, mu, X_obs)

plt.figure(2)
map=plot_map(longgridplot, latgridplot, np.sqrt(V), X_obs, levels=np.arange(0,np.sqrt(V).max()+1,0.25))

#plt.show()




###################################################################################
#
#
# Repeat but now with Spherical kernel
#
#
###################################################################################



k2 = Exponentialhaversine(2, lengthscale=2000)
m2 = GPy.models.GPRegression(X_obs, y_obs,k2)
m2.optimize_restarts(10)
print(m2)
mu2,V2 = m2.predict(X_plot)

plt.figure(1)
map=plot_map(longgridplot, latgridplot, mu2, X_obs)
plt.title('Spherical GP - constant error')

plt.figure(2)
map=plot_map(longgridplot, latgridplot, V2, X_obs, levels=np.arange(0,V2.max()+1,0.25))
#plt.show()



###################################################################################
#
#
# Repeat but now with Spherical kernel and heteroscedastic likelihood.
#
#
###################################################################################

from scaledheteroscedasticgaussian import ScaledHeteroscedasticGaussian
from gp_heteroscedastic_ratios import ScaledHeteroscedasticRegression


k3 = Exponentialhaversine(2, lengthscale=2000)
m3 = ScaledHeteroscedasticRegression(X=X_obs, Y=y_obs, kernel=k3, noise_mult=1., known_variances=var_ratios)
m3.optimize_restarts(10)
print(m3)

mu3,V3 = m3.predict_noiseless(X_plot)


plt.figure(3)
map=plot_map(longgridplot, latgridplot, mu3, X_obs)
plt.title('Spherical GP - heteroscedastic error')

plt.figure(4)
map=plot_map(longgridplot, latgridplot, V3, X_obs, levels=np.arange(0,V3.max()+1,0.25))
#plt.show()


############################################################
#
# Let's now evaluate the log-likelihood
#
#
############################################################

from scipy.stats import multivariate_normal

count=0
loglikes = np.zeros(len(gcm_SSTs))

#
#tmp = np.column_stack((gcm_grid, gcm_grid))
#multivariate_normal.logpdf(tmp.T, mu.flatten(), Cov)
# Is quicker

# Best we can hope for is...

from Cholesky import *

multivariate_normal.logpdf(mu.flatten(), mu.flatten(), Cov)

# What else can we compare with?

for file_name in gcm_SSTs:

    file_nm = file_name.split(GCM_dir)[-1]
    print(file_nm)

    # Read in GCM output.
    gcm1 = np.genfromtxt(file_name)


    X_pred, gcm_grid = ThinGrid(gcm1, gcm_mask, thinby=5)
    mu, Cov = m3.predict_noiseless(X_pred, full_cov=True)

    loglikes[count] = multivariate_normal.logpdf(gcm_grid.flatten(), mu.flatten(), Cov)

    # what should I do about the measurement error?
    # Or is there no measurement error as we've modelled the underlying surface with a GP!
    count += 1



"""
To do
- Check sanity of loglike values
- why do random values have much smaller values than the simulations?
- do the tests we discussed
- describe problem with ARD kernel.
- robust scoring rules?
- compare to MSE
- check likelihood calculation - seems to not be robust to thinning of the grid.

"""


"""

k._unscaled_dist(X_obs)

###################
k1 = Exponentialhaversine(2, lengthscale=2000)
# how do we specify a different noice variance at each locations

m = GPy.models.GPRegression(X_obs, y_obs, k1)
m.optimize_restarts(10)
print(m)


#########################
#
#  Predict at GCM grid locations.
#


lats = np.arange(-89.375,89.375,1.25)
longs = np.arange(-180,178.75+0.1, 1.25)
longgrid, latgrid = np.meshgrid(longs, lats)

GCM_dir = os.getenv("HOME")+'/Documents/Code/ModelDataComparison/DMC/Scripts/Model_data/CO2_anom/'
file_name = 'tczyi.txt'
gcm1 = np.genfromtxt(GCM_dir+file_name)
gcm_mask = np.genfromtxt(GCM_dir+'mask.txt', dtype='int')

Xpred = np.column_stack((longgrid.flatten()[gcm_mask-1], latgrid.flatten()[gcm_mask-1]))
ypred, Vpred = m.predict_noiseless(Xpred)

yplot = np.zeros(longgrid.size)
yplot[gcm_mask-1] = ypred

############### Equidistant Cylindrical Projection ####################################
# The simplest projection, just displays the world in latitude/longitude coordinates.
m = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,\
            llcrnrlon=-180,urcrnrlon=180,resolution='c')
m.drawcoastlines()
#m.fillcontinents(color='coral',lake_color='aqua')
# draw parallels and meridians.
m.drawparallels(np.arange(-90.,91.,30.))
m.drawmeridians(np.arange(-180.,181.,60.))
m.drawmapboundary(fill_color='aqua')
plt.xlabel('lon')
plt.ylabel('lat')
levels = np.arange(-10,10,1)
from cmocean import cm as cm
m.contourf(longgrid,latgrid,yplot.reshape(lats.size,longs.size),15,levels=levels,
    cm = cm.thermal, linewidths=1.5)
m.colorbar()
m.scatter(X_obs[:,0],X_obs[:,1] )
plt.show()
"""
