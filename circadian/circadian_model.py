# Copyright (c) 2018 Pablo Moreno-Munoz
# Universidad Carlos III de Madrid and eB2 (eb2.tech)

import numpy as np
import paramz
import util
from maximization import MaximizationStep
from expectation import ExpectationStep
import random
import matplotlib.pyplot as plt

class CircadianModel(paramz.Model):
    def __init__(self, Y, K, T, C=None, var_precision=None, name='Circadian'):
        super(CircadianModel, self).__init__(name=name)

        self.Y = Y
        self.N = Y.shape[0]
        self.K = K

        # automatic set-up #####
        if T is None:
            self.T = 24 # hourly precision
        else:
            self.T = T

        assert self.T == Y.shape[1]

        if C is None:
            self.C = 3
        else:
            self.C = C

        if var_precision is None:
            self.var_precision = 2
            #self.var_precision = 1
        else:
            self.var_precision = var_precision
        ###########################

        self.t = np.arange(0, 1, 1 / self.T)[:, np.newaxis]

        self.maximization_step = MaximizationStep()
        self.expectation_step = ExpectationStep()

        # Parameters
        self.pi = paramz.Param('pi', np.random.dirichlet(50*np.ones((1, self.K)).flatten(), 1))
        self.mu = paramz.Param('mu', 0.5*np.ones((self.T, self.K)))
        self.f = paramz.Param('f', np.random.randn(self.T, self.K))

        # Kernel hyperparameters
        self.l = paramz.Param('lengthscale', 0.01*np.ones((1, self.K)))
        self.a0 = paramz.Param('a0_fourier', np.random.randn(1, self.K))
        self.a = paramz.Param('a_fourier', np.random.randn(self.C, self.K))
        self.b = paramz.Param('b_fourier', np.random.randn(self. C, self.K))
        self.sigmas = paramz.Param('sigmas', np.ones((self.var_precision, 1)))

        # Parameters:
        self.model_parameters = [self.pi, self.f, self.mu]

        # Hyperparameters:
        self.link_parameter(self.l, index=0)
        self.link_parameter(self.a0)
        self.link_parameter(self.a)
        self.link_parameter(self.b)
        self.link_parameter(self.sigmas)

        # Expected variables
        self.r_ik, self.c_ik, self.Y_exp, self.matrices = self.expectation_step.expectation(Y=self.Y, K=self.K, C=self.C, t=self.t,
                                                                             pi=self.pi,
                                                                             parameters=self.model_parameters,
                                                                             hyperparameters=self.parameters)

        self.expected_values = {'r_ik': self.r_ik, 'c_ik': self.c_ik, 'Y_exp': self.Y_exp, 'matrices': self.matrices}

    def objective_function(self):
        return -self._log_likelihood

    def parameters_changed(self):
        self._log_likelihood, gradients = self.maximization_step.maximization(Y=self.Y, K=self.K, C=self.C,
                                                                                                  t=self.t, parameters=self.model_parameters,
                                                                                                  hyperparameters=self.parameters,  #los parameters son los linked
                                                                                                  expectations=self.expected_values)

        # Loading gradients:
        self.l.gradient = -gradients['dL_dl']
        self.a0.gradient= -gradients['dL_da0']
        self.a.gradient = -gradients['dL_da']
        self.b.gradient = -gradients['dL_db']
        self.sigmas.gradient = -gradients['dL_dsigmas']   #por que pone self.b.sigmas = ... --- self.sigmas.gradient

    def em_algorithm(self, iterations, convergence=False):

        for it in range(iterations):
            # E-Step
            self.r_ik, self.c_ik, self.Y_exp, self.matrices = self.expectation_step.expectation(Y=self.Y, K=self.K, C=self.C, t=self.t, pi=self.pi,
                                                                  parameters=self.model_parameters,
                                                                  hyperparameters=self.parameters)

            self.expected_values['r_ik'] = self.r_ik
            self.expected_values['c_ik'] = self.c_ik
            self.expected_values['Y_exp'] = self.Y_exp
            self.expected_values['matrices'] = self.matrices

            # M-Step
            for k in range(self.K):
                Y_exp_real = self.Y_exp[k][:,:,0]
                Y_exp_bin = self.Y_exp[k][:, :, 1]

                self.pi[0, k] = sum(self.r_ik[:, k]) / self.N
                self.f[:, k] = sum(Y_exp_real*np.tile(self.r_ik[:,k,np.newaxis],(1, self.T))/sum(self.r_ik[:,k]),0)
                self.mu[:,k] = sum(Y_exp_bin*np.tile(self.r_ik[:,k,np.newaxis],(1, self.T))/sum(self.r_ik[:,k]),0)

            self.optimize(messages=True, max_iters=100)