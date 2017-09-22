
"""
This code demonstrates how to use GPy's heteroscedastic error models. See Test_scaledheteroscedastic
for my code.
"""

import GPy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pylab as pb

xlocs = np.arange(-10,10,2)
X = np.repeat(xlocs,10)
Xvars = np.repeat(xlocs**2/10.,10)+1
y = X + np.random.normal(0, Xvars)
X = X.reshape(-1,1)
y = y.reshape(-1,1)
plt.ion()
plt.scatter(X, y)
#plt.show()


Y_metadata = {'output_index': Xvars}

k = GPy.kern.RBF(1)
m2 = GPy.models.GPHeteroscedasticRegression(X, y, k)# Y_metadata=Y_metadata)
m2['.*het_Gauss.variance'] = Xvars[:,None]
m2.het_Gauss.variance.fix()

Xplot = np.linspace(-15,15,100)
mu, V = m2.predict_noiseless(Xplot.reshape(-1,1), full_cov=True)
samples = np.random.multivariate_normal(mu.flatten(),V, 100)




ax = m2.plot_f() #Show the predictive values of the GP.
for i in range(100):
    plt.plot(Xplot, samples[i,:], linestyle=':', linewidth=0.5, color='gray')
plt.plot(Xplot, mu, 'k-', linewidth=3)
plt.scatter(X,y)

ax.plot([-15,15], [-15,15], ls="--", c=".3", color='red')
