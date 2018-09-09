# Copyright (c) 2018 Pablo Moreno-Munoz
# Universidad Carlos III de Madrid and eB2 (eb2.tech)

import numpy as np
import paramz
import util
from GPy.util import linalg

class ExpectationStep():

    def expectation(self, Y, K, C, t, pi, parameters, hyperparameters):

        self.N = Y.shape[0]
        self.T = Y.shape[1]

        # Model parameters
        pi = parameters[0].copy()
        f = parameters[1].copy()
        mu = parameters[2].copy()

        # Model hyperparameters
        ls = hyperparameters[0].copy()
        a0 = hyperparameters[1].copy()
        a = hyperparameters[2].copy()
        b = hyperparameters[3].copy()
        sigmas = hyperparameters[4].copy()

        # Missing values
        Yreal = Y[:, :, 0]
        Ybin = Y[:, :, 1]

        nans = np.isnan(Yreal)
        notnans = np.invert(nans)

        # Covariance
        hyperparam_list = [ls, a0, a, b, sigmas]
        S, L, Si = util.build_covariance(t, K, hyperparam_list) #dims: (T,T,K)
        matrices = {'S_old': S, 'L_old': L, 'Si_old': Si}

        # Posterior of latent classes
        r_ik = np.empty((self.N, K))

        # Expectations on latent variables
        for k in range(K):
            #param_list = [f[:,k], S[:,:,k], Si[:,:,k], mu[:,k]]
            S_k = S[:, :, k]
            Si_k = Si[:, :, k]
            mu_k = mu[:,k]
            detS_k = np.linalg.det(S_k)
            #r_ik[:, k] = pi[0, k] * util.heterogeneous_pdf(Y, param_list)
            for i in range(self.N):
                r_ik[i,k] = pi[0,k]*(1/np.sqrt(detS_k * np.pi**self.T)) \
                            *np.exp(-0.5*np.dot(Yreal[i,np.ix_(notnans[i,:])], Si_k[np.ix_(notnans[i,:],notnans[i,:])]).dot(Yreal[i,np.ix_(notnans[i,:])].T)) \
                            *np.prod((mu_k[np.ix_(notnans[i,:])])**Ybin[i,np.ix_(notnans[i,:])]  * (1 - mu_k[np.ix_(notnans[i,:])])**Ybin[i,np.ix_(notnans[i,:])])

        r_ik = r_ik/np.tile(r_ik.sum(1)[:,np.newaxis],(1,K))

        # Expectations on missing values
        c_ik = np.empty((self.N, K))
        Y_expectation = []

        for k in range(K):
            # Real observations
            Yreal_fill = Yreal.copy()
            S_k = S[:, :, k]
            Si_k = Si[:, :, k]

            for i in range(self.N):
                S_k_oo = S_k[np.ix_(notnans[i,:], notnans[i,:])]
                S_k_mm = S_k[np.ix_(nans[i,:], nans[i,:])]
                S_k_mo = S_k[np.ix_(nans[i,:], notnans[i,:])]
                S_k_om = S_k_mo.T
                Si_k_mm = Si_k[np.ix_(nans[i,:], nans[i,:])] # mm submatrix of Si_k

                L_k_oo = linalg.jitchol(S_k_oo)
                iS_k_oo, _ = linalg.dpotri(np.asfortranarray(L_k_oo)) # inverse of oo submatrix

                Cov_m = S_k_mm - (S_k_mo.dot(iS_k_oo).dot(S_k_om))
                c_ik[i,k] = np.trace(Si_k_mm.dot(Cov_m))

                Yreal_fill[i, nans[i, :]] = S_k_mo.dot(iS_k_oo).dot(Yreal[i, notnans[i, :]])


            # Binary observations
            Ybin_fill = Ybin.copy()
            mu_matrix = np.tile(mu[:,k].T + 0.0,(self.N,1))
            Ybin_fill[nans] = mu_matrix[nans]

            # Missings observation are now filled
            Y_fill_k = np.empty((self.N, self.T, 2))
            Y_fill_k[:,:,0] = Yreal_fill
            Y_fill_k[:,:,1] = Ybin_fill

            Y_expectation.append(Y_fill_k)

        return r_ik, c_ik, Y_expectation, matrices