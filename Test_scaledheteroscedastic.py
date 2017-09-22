import GPy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from scaledheteroscedasticgaussian import ScaledHeteroscedasticGaussian
from gp_heteroscedastic_ratios import ScaledHeteroscedasticRegression

#
# checked by comparing with GPy when all known_variances=1
#
k = GPy.kern.RBF(1, lengthscale=2.0)

X = np.linspace(0, 10, 20)[:, None]
K = k.K(X)
f = np.random.multivariate_normal(mean=np.zeros_like(X).flatten(), cov=K)[:, None]
orig_X = X
orig_f = f

known_ratios = np.abs(X)**2 / 10
known_scale = 0.3 / 1.0
true_variances = known_ratios * known_scale
samples_per_location = 5

X = np.repeat(X,samples_per_location)[:,None]
f = np.repeat(f,samples_per_location)[:, None]
stacked_true_variances = np.repeat(true_variances,samples_per_location)
stacked_known_ratios = np.repeat(known_ratios, samples_per_location)[:, None]

Y = f + (np.random.randn(f.shape[0])*np.sqrt(stacked_true_variances.flatten()))[:, None]
#Y2 = f + (np.random.randn(f.shape[0])*np.sqrt(true_variances.flatten()))[:, None]
#Y3 = f + (np.random.randn(f.shape[0])*np.sqrt(true_variances.flatten()))[:, None]
#all_known_ratios = np.vstack((known_ratios, known_ratios, known_ratios))
#Y = np.vstack((Y1, Y2, Y3))
#X = np.vstack((X, X, X))

 #plt.plot(X,Y, 'rx')
#plt.plot(orig_X, orig_f, 'g')
#plt.show()


k1 = GPy.kern.RBF(1)
m2 = ScaledHeteroscedasticRegression(X=X, Y=Y, kernel=k1, noise_mult=0.1, known_variances=stacked_known_ratios)

m2.likelihood.checkgrad(verbose=1)

m2.optimize()

k2 = GPy.kern.RBF(1)
#m =GPy.models.GPRegression(X,y,k)
#print(m)
m1 = GPy.models.GPRegression(X=X, Y=Y, kernel=k2)
m1.likelihood.variance = (stacked_known_ratios*0.1).mean()
m1.optimize()


def plot_samples(m):
    #m2.Scaled_het_Gauss.variance.fix()
    Xplot = np.linspace(0,10,100)
    mu, V = m.predict_noiseless(Xplot.reshape(-1,1), full_cov=True)
    samples = np.random.multivariate_normal(mu.flatten(),V, 50)

    for i in range(samples.shape[0]):
        plt.plot(Xplot, samples[i,:])

    plt.plot(X, Y, 'ro', lw=10)
    plt.plot(orig_X, orig_f, 'g', lw=5)

plt.figure()
plot_samples(m1)
plt.figure()
plot_samples(m2)
plt.show()


"""
actual_variance = 0.5
samples_per_location = 30
##
xlocs = np.arange(-10,10,2)
X = np.repeat(xlocs,samples_per_location)
#X = np.linspace(-10,10,100)
Xvars = np.repeat(xlocs**2/10.,samples_per_location)+1
Xvars = Xvars[:,None]
X = X.reshape(-1,1)
y = X**2 + np.random.randn(*Xvars.shape)*np.sqrt(Xvars)  # multiply by noise_mult
#y = X + np.random.randn(*X.shape)*np.sqrt(actual_variance)

p
#plt.ion()
#plt.scatter(X, y)
#plt.show()

k = GPy.kern.RBF(1)
#m =GPy.models.GPRegression(X,y,k)
#print(m)
m1 = GPy.models.GPRegression(X=X, Y=y, kernel=k)
m1.likelihood.variance = 2.0

k = GPy.kern.RBF(1)
m2 = ScaledHeteroscedasticRegression(X=X, Y=y, kernel=k, noise_mult=3.5, known_variances=Xvars)

m2.likelihood.checkgrad(verbose=1)
#m2.Scaled_het_Gauss.variance.fix()

print(m2)


### need to fix


Xplot = np.linspace(-15,15,100)
mu, V = m2.predict_noiseless(Xplot.reshape(-1,1), full_cov=True)
samples = np.random.multivariate_normal(mu.flatten(),V, 100)

"""
