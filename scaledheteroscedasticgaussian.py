# Copyright (c) 2012-2014 The GPy authors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)
#TODO
"""

"""

import numpy as np
from scipy import stats, special
from GPy.likelihoods import link_functions
#from .likelihood import Likelihood
from GPy.core.parameterization import Param
from paramz.transformations import Logexp
from scipy import stats
from GPy.likelihoods import Gaussian


class ScaledHeteroscedasticGaussian(Gaussian):
    def __init__(self, Y_metadata, gp_link=None, noise_mult=1., known_variances=1., name='Scaled_het_Gauss'):
        if gp_link is None:
            gp_link = link_functions.Identity()

        if not isinstance(gp_link, link_functions.Identity):
            print("Warning, Exact inference is not implemeted for non-identity link functions,\
            if you are not already, ensure Laplace inference_method is used")

        # note the known_variances are fixed, not parameterse
        self.known_variances = known_variances
        self.noise_mult = Param('noise_mult', noise_mult, Logexp()) # Logexp ensures its positive
        # this is a parameter, so it gets optimized, gradients calculated etc.

        #super(ScaledHeteroscedasticGaussian, self).__init__(gp_link, variance=1.0, name=name)
        super(Gaussian, self).__init__(gp_link, name=name)
        # note: we're inheriting from Likelihood here, not Gaussian, so as to avoid problems with the Gaussian variance.

        #add a new parameter by linking it (see just above in GPy.likelihoods.gaussian.Gaussian).
        self.link_parameter(self.noise_mult)

        if isinstance(gp_link, link_functions.Identity):
            self.log_concave = True

    # rather than having an object self.variance, we define it to be a function that is called whenever variances is needed
    @property
    def variance(self):
        return self.known_variances*self.noise_mult

#compute the gradients,
# dL_dalpha and add them in,
#def exact_inference_gradients(self, dL_dKdiag,Y_metadata=None)
# this might take a little thinking

    def exact_inference_gradients(self, dL_dKdiag,Y_metadata=None):
        grad = np.sum(dL_dKdiag[:, np.newaxis]*self.known_variances)
        # see page 114, section 5.4 in Rasmussen and Williams
        #dL/dnoise_mult = tr(dL/dK * dK/dnoise_mult)
        # dL/dK not changed
        # dK_dnoisemult = I(*h*)v as K = K + noise_mult * I*v where I (*h*) v  is hadamard product
        #print("dL_dKdiag dim: {} known_variance_dim: {} overall_dim: {}".format(dL_dKdiag.shape, self.known_variances.shape, (dL_dKdiag*self.known_variances).shape))
        return grad
## COMPLETE GUESS


# update the gradients of the parameters,
    def update_gradients(self, grad):
        self.noise_mult.gradient = 0 # needed or not?
        self.noise_mult.gradient = grad
#  for example, depending on how we choose to return things in exact_inference_gradients


    def gaussian_variance(self, Y_metadata=None):
        return self.variance

# override gaussian_variance(self, Y_metadata=None) as self.alpha*self.variance[Y_metadata['output_index'].flatten()]

    def predictive_values(self, mu, var, full_cov=False, Y_metadata=None):
        print("Warning: prediction of values not implemented")

    def predictive_quantiles(self, mu, var, quantiles, Y_metadata=None):
        print("Warning: predictions of values not implemented")

    def samples(self, gp, Y_metadata=None):
        print("Warning: samples not implemented")

    def variational_expectations(self, Y, m, v, gh_points=None, Y_metadata=None):
        print("Warning: variational_expectations not implemented")
