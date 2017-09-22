# Copyright (c) 2012 - 2014 the GPy Austhors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from GPy.core import GP
from GPy import likelihoods
from GPy import kern
from GPy import util
from scaledheteroscedasticgaussian import ScaledHeteroscedasticGaussian

class ScaledHeteroscedasticRegression(GP):
    """
    Gaussian Process model for heteroscedastic regression
    Assumes that the observation noise is known up to a multiplicative ratio only
    This is a thin wrapper around the models.GP class, with a set of sensible defaults

    :param X: input observations
    :param Y: observed values
    :param kernel: a GPy kernel, defaults to rbf
    :param noise_mult: variance is noise_nult*diag(known_variances)
    :param known_variances: the ratio of the variances. The shape should match that of Y.
    
    NB: This model does not make inference on the noise outside the training set
    """
    def __init__(self, X, Y, kernel=None, Y_metadata=None, noise_mult=1., known_variances=1.0):

        if kernel is None:
            kernel = kern.RBF(X.shape[1])

        assert known_variances.shape == Y.shape
        #Likelihood
        likelihood = ScaledHeteroscedasticGaussian(Y_metadata = Y_metadata, noise_mult=noise_mult, known_variances=known_variances)

        super(ScaledHeteroscedasticRegression, self).__init__(X,Y,kernel,likelihood, Y_metadata=Y_metadata)
