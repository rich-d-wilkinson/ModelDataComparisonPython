
"""
Code for calculating log pdf and simulating multivariate Gaussians from the
Cholesky decomposition of Sigma. The idea is to only have to compute this once.
"""

import numpy as np
import scipy.linalg as slin




# tests
"""
import GPy

x = np.linspace(0,5,10)[:,None]
k = GPy.kern.RBF(1)
Sigma = k.K(x)

Chol = np.linalg.cholesky(Sigma)

# Determinant
np.linalg.det(Sigma)
np.prod(np.diag(Chol))**2

# Random variables
"""


def rmvnorm(n, mu, Chol):
    """
    n is number of replicates
    mu is d * 1 mean vector
    Chol is d*d Cholesky decomposition of the covariance matrix

    Returns a d * n matrix of samples (one per column)
    """

    z = np.random.normal(size=(mu.size,n))
    return mu + np.dot(Chol,z)

"""
tmp = rmvnorm(1000, np.arange(10)[:,None], Chol)
tmp.mean(axis=1)
np.cov(tmp)-Sigma
####################
"""

def dlogmvnorm(x, mu, Chol):
    """
    Efficiently compute the log of the Gaussian pdf from the Cholesky decomposition

    x is a d * n vector of values where we want to compute the log pdf.
    mu is d * 1 mean vector
    Chol is d*d Cholesky decomposition of the covariance matrix

    Returns a 1*n vector of log pdfs.
    """
    assert x.shape[0]==mu.shape[0]
    assert x.ndim==mu.ndim==Chol.ndim==2
    assert mu.shape[1]==1
    assert Chol.shape[0] == mu.shape[0]
    assert Chol.shape[1] == mu.shape[0]

    xc = x-mu
    y = slin.solve_triangular(Chol, xc, lower=True)
    exponent  = -0.5* np.diag(np.dot(y.T,y))
#    logdet = -np.log(np.prod(np.diag(Chol)))
    logdet = -np.sum(np.log(np.diag(Chol)))
    # -0.5*np.log(np.prod(np.diag(Chol))**2)
    return(exponent + logdet - mu.size/2.*np.log(2*np.pi))

"""
x=tmp[:,:1]   #tmp[:,0][:,None]
mu = np.arange(10)[:,None]
xc = x-mu

np.dot(xc.T, np.dot(np.linalg.inv(Sigma),xc))
y = slin.solve_triangular(Chol, xc, lower=True)
-0.5*np.diag(np.dot(y.T,y))

np.log(1/np.sqrt((2*np.pi)**mu.size))
-mu.size/2.*np.log(2*np.pi)


np.log(1/np.sqrt(np.linalg.det(Sigma)))
-0.5*np.log(np.prod(np.diag(Chol))**2)



from scipy.stats import multivariate_normal
print(multivariate_normal.logpdf(x.flatten(), mu.flatten(), Sigma))
print(dlogmvnorm(x[:,None], mu, Chol))

for ii in range(10):
    print(multivariate_normal.logpdf(tmp[:,ii].flatten(), mu.flatten(), Sigma))


dlogmvnorm(tmp[:,:1], mu, Sigma)
x=tmp[:,0][:,None]
dlogmvnorm(tmp[:,:10], mu, Chol)
multivariate_normal.logpdf(x.flatten(), mu.flatten(), Sigma)
"""
