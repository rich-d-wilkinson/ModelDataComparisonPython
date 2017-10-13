
# coding: utf-8

# # Quick Demo
# I'll use this doc to briefly demo the code. The core functionality to fit the GP and evaluate the log-likelihood is just six lines of code (3 lines to train the GP, 3 lines to calculate the likelihood). Of course we need to load the libraries, data, grids, do plots, print results etc, so that what the rest of the code here is doing.
# 
# Let's begin by loading the libraries we'll need.

# In[1]:

import GPy
import sys
import os
sys.path.append(os.getenv("HOME") + "/Documents/Code/Emulation/GPyDifferentMetrics/")
from HaversineDist import Exponentialhaversine
import numpy as np
get_ipython().magic('matplotlib inline')
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib
import glob as glob
from cmocean import cm as cm
from Utilities import *


# Now let's specify where the data and model runs are located, and load them.

# In[2]:


GCM_dir = os.getenv("HOME")+'/Documents/Code/ModelDataComparison/DMC/Scripts/Model_data/CO2_anom/'
gcm_SSTs = glob.glob(GCM_dir+'t*.txt')

gcm_mask = np.genfromtxt(GCM_dir+'mask.txt', dtype='int')

obs_dir = os.getenv("HOME")+'/Documents/Code/ModelDataComparison/DMC/Scripts/Observation_data/P3+_SST_anom/'
file = 'lambda_10.txt'
observations = np.genfromtxt(obs_dir+file, skip_header=1)


# We'll extract the observations into their coordinates (long and lat) which I'll refer to as X, the measurement (Y), and our estimate of the ratio of the variance of the measurement error at each point (NOTE: I was unsure whether the numbers in the data you gave me were supposed to standard deviations or variances - they look like std devs, so I've squared them). Note that all that matters is the relative size of these variances (their ratio). We need to try changing this to find its effect.
# 
# I've plotted the locations just as a check.

# In[3]:

X_obs = observations[:,0:2]
y_obs = observations[:,2].reshape(-1,1)
var_ratios = observations[:,3][:,None]**2
map = plot_map(X_obs=X_obs)


# ## Fitting a Gaussian process model
# 
# 

# In[4]:


from scaledheteroscedasticgaussian import ScaledHeteroscedasticGaussian
from gp_heteroscedastic_ratios import ScaledHeteroscedasticRegression


k3 = Exponentialhaversine(2, lengthscale=2000)
m3 = ScaledHeteroscedasticRegression(X=X_obs, Y=y_obs, kernel=k3, noise_mult=1., 
                                     known_variances=var_ratios)
m3.optimize_restarts(10)


# Note that the fitting the GP is just three lines of code. We can visualise the model fit by predicting on a grid.

# In[5]:

print(m3)
latsplot = np.arange(-90.0,90.0, 2.5)
longsplot = np.arange(-180.0,180.0, 2.5)
longgridplot, latgridplot = np.meshgrid(longsplot, latsplot)
X_plot=np.column_stack((longgridplot.flatten(), latgridplot.flatten())) # specifies the prediction locations

mu3,V3 = m3.predict_noiseless(X_plot)


plt.figure(3)
map=plot_map(longgridplot, latgridplot, mu3, X_obs)
plt.title('Predict mean SST anomaly - spherical GP, heteroscedastic error')

plt.figure(4)
map=plot_map(longgridplot, latgridplot, np.sqrt(V3), X_obs, levels=np.arange(0,np.sqrt(V3).max()+1,0.25))
plt.title('Standard deviation of the prediction')


# ## Likelihood calculations
# 
# Let's now evaluate the 8 GCM simulations we have available. To do this, we need to predict the GCM output at every grid cell, and then evalaute the probability of seeing this value under our GP model fitted to the observational data.
# 
# We'll begin by loading the GCM runs

# In[6]:

count=0
gcm_runs = np.zeros((8,27186))
gcm_runs_label = gcm_SSTs.copy()

for file_name in gcm_SSTs:
    file_nm = file_name.split(GCM_dir)[-1]
    print(file_nm)
    # Read in GCM output.
    gcm_runs[count,:] = np.genfromtxt(file_name)
    gcm_runs_label[count] = file_nm.split(".txt")[0]
    count +=1

    
# Create the prediction grid - removing the land coordinates to save computation effort.
X_pred, out = ThinGrid(gcm_runs[0,:], gcm_mask, thinby=1)


# Let's predict on the grid, and compute the Cholesky decomposition of the covariance matrix.

# In[7]:

mu, Cov = m3.predict_noiseless(X_pred, full_cov=True)
Chol = np.linalg.cholesky(Cov)


# In[8]:

from Cholesky import *
loglikes = dlogmvnorm(gcm_runs.T, mu, Chol)
orderings = np.argsort(-loglikes) #minus sign so that max is first
relative = np.round(loglikes-np.max(loglikes),1)


# Note that the likelihood calculation is just three lines (once we've defined a grid).
# 
# We can decode from the file names to the CO2 values to this to see how this ranks the various GCM runs.

# In[9]:

dict = {'tdgth': '280 ppm',
        'tczyi': '315 ppm',
    'tdgtj': '350 ppm',
    'tczyj': '375 ppm',
    'tdgtg': '405 ppm',
    'tczyk': '475 ppm',
    'tdgtk': '560 ppm',
    'tdgti': '1000 ppm'}

for ii in range(8):
    print(dict[gcm_runs_label[orderings[ii]]] + ':  relative loglike = '+str(relative[orderings[ii]]))


# Let's plot the GCM runs in order of their log-likelihood ranking.

# In[10]:

for ii in range(8):
    plt.figure()
    map = plot_gcm(gcm_runs[orderings[ii],:], gcm_mask)
    plt.title(dict[gcm_runs_label[orderings[ii]]]+ ':  loglike = '+ str(loglikes[orderings[ii]]))
#    plt.title(loglikes[ii]+': log-likelihood = ' + str(round(float(loglike_results[ii,0]),1)))


# We can compare this with the ranking we obtain using the RMSE as a score function. 

# In[11]:


def MSE(X_obs, X_pred, y_obs, gcm_out):
    k = Exponentialhaversine(2)
    index = np.argmin(k._unscaled_dist(X_obs, X_pred), axis=1)
    #print(k._unscaled_dist(X_obs, X_pred)[range(y_obs.size),index])
    y_gcm_grid = gcm_out[index]
    #print(y_gcm_grid)
    #print(X_pred[index])
    return(np.sqrt(np.mean((y_gcm_grid-y_obs)**2))) 

X_pred, out = ThinGrid(gcm_runs[0,:], gcm_mask, thinby=1)

MSEs = np.zeros(8)
for ii in range(8):
    MSEs[ii]=(MSE(X_obs, X_pred, y_obs, gcm_runs[ii,:]))
MSEorderings = np.argsort(MSEs)
print(MSEorderings)


# In[12]:

X_obs.shape



# In[13]:

CO2 = np.zeros(8)
for ii in range(8):
    CO2[ii] = int(dict[gcm_runs_label[ii]].split(' ppm')[0])
    
plt.scatter(CO2, -MSEs)

plt.title('RMSE vs pCO2')


# In[14]:

plt.figure()
plt.scatter(CO2, loglikes)
plt.title('log-likelihood vs pCO2')


# So you can see we get a different ranking from the log-likelihood and the RMSE score. In a sense, this is what we hoped for. Here, the data are telling us smaller pCO2s are sensible. The RMSE tells us pCO2 $\approx$450
#  ppm is best.
# 
# Curiously, our RMSE results don't match Fran's RMSE results. This may be because I'm using a daft pairing of data and simulation.

# ##### 
