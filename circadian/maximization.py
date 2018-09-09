# Copyright (c) 2018 Pablo Moreno-Munoz
# Universidad Carlos III de Madrid and eB2 (eb2.tech)

import numpy as np
import paramz
import util
from GPy.util import linalg

class MaximizationStep():

    def maximization(self, Y, K, C, t, parameters, hyperparameters, expectations):

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

        var_precision = sigmas.shape[0]

        # Expected values
        r_ik = expectations['r_ik']
        #c_ik = expectations['c_ik']
        Y_exp = expectations['Y_exp']
        matrices = expectations['matrices']

        # old building of matrices
        Sold = matrices['S_old']
        Lold = matrices['L_old']
        Siold = matrices['Si_old']

        # new building of matrices
        hyperparam_list = [ls, a0, a, b, sigmas]
        S, L, Si = util.build_covariance(t, K, hyperparam_list) #dims: (T,T,K)

        # Identifiying missing (NaN) values
        nans = np.isnan(Y[:,:,0])
        notnans = np.invert(nans)

        # Expected Log-Likelihood (Cost Function)
        log_likelihood = 0.0
        het_logpdf = np.empty((self.N, K))

        # Log-likelihood derivatives wrt hyperparameters
        dL_dl = np.zeros((1, K))
        dL_da0 = np.zeros((1, K))
        dL_da = np.zeros((C, K))
        dL_db = np.zeros((C, K))
        dL_dsigmas = np.zeros((var_precision, 1))

        c_ik = np.empty((self.N, K))

        for k in range(K):
            S_k = S[:, :, k] # new
            Si_k = Si[:, :, k] # new

            Sold_k = Sold[:, :, k] # old
            Siold_k = Siold[:, :, k] # old

            Y_exp_k = Y_exp[k]
            Y_exp_real = Y_exp_k[:, :, 0]
            Y_exp_bin = Y_exp_k[:, :, 1]
            detS_k = np.linalg.det(S_k)

            for i in range(self.N):
                Sold_k_oo = Sold_k[np.ix_(notnans[i,:], notnans[i,:])]
                Sold_k_mm = Sold_k[np.ix_(nans[i,:], nans[i,:])]
                Sold_k_mo = Sold_k[np.ix_(nans[i,:], notnans[i,:])]
                Sold_k_om = Sold_k_mo.T
                Si_k_mm = Si_k[np.ix_(nans[i,:], nans[i,:])] # mm submatrix of Si_k

                Lold_k_oo = linalg.jitchol(Sold_k_oo)
                iSold_k_oo, _ = linalg.dpotri(np.asfortranarray(Lold_k_oo)) # inverse of oo submatrix

                Cov_m = Sold_k_mm - (Sold_k_mo.dot(iSold_k_oo).dot(Sold_k_om))
                c_ik[i,k] = np.trace(Si_k_mm.dot(Cov_m))

                A_m = np.zeros((self.T, self.T))
                A_m[np.ix_(nans[i, :], nans[i, :])] = Cov_m

                y = Y_exp_real[i, :].T
                y = y[:, np.newaxis]
                yy_T = np.dot(y,y.T)
                aa_T = Si_k.dot(yy_T).dot(Si_k.T)

                Q1 = aa_T - Si_k
                Q2 = Si_k.dot(A_m).dot(Si_k)

                dK_dl, dK_da0, dK_da, dK_db, dK_dsigmas = self.kernel_gradients(Q1, Q2, t, k, C, hyperparam_list)


                dL_dl[0,k] += 0.5*r_ik[i,k]*dK_dl
                dL_da0[0, k] += 0.5*r_ik[i,k]*dK_da0
                dL_da[:,k] += 0.5*r_ik[i,k]*dK_da.flatten()
                dL_db[:,k] += 0.5*r_ik[i,k]*dK_db.flatten()
                dL_dsigmas += 0.5*r_ik[i,k]*dK_dsigmas

                log_likelihood += - 0.5*r_ik[i,k]*np.log(pi[0,k]) - 0.5*r_ik[i,k]*np.log(detS_k) \
                                  - 0.5*r_ik[i,k] * np.dot(Y_exp_real[i,:],Si_k).dot(Y_exp_real[i,:].T) \
                                  - 0.5*r_ik[i,k]*c_ik[i,k] \
                                  + r_ik[i,k]*np.sum(Y_exp_bin[i,:]*np.log(mu[:, k])) \
                                  + r_ik[i,k]*np.sum(Y_exp_bin[i,:]*np.log(1 - mu[:, k]))
                                # + r_ik[i,k]*[]
                                # falta el pi de la gaussian


            #param_list = [f[:, k], S[:, :, k], Si[:, :, k], mu[:, k]]

        gradients = {'dL_dl':dL_dl, 'dL_da0':dL_da0, 'dL_da':dL_da, 'dL_db':dL_db, 'dL_dsigmas':dL_dsigmas}

        return log_likelihood, gradients

    def kernel_gradients(self, Q1, Q2, t, k, C, hyperparam_list):

        # Kernel gradients wrt hyperparameters
        ls = hyperparam_list[0].copy()
        a0 = hyperparam_list[1].copy()
        a = hyperparam_list[2].copy()
        b = hyperparam_list[3].copy()
        sigmas = hyperparam_list[4].copy()

        n_fourier = a.shape[0]
        var_precision = sigmas.shape[0]

        trig_arg = np.pi * (t - t.T) / self.T

        hyperparam_k_list = [ls[0, k], a0[0, k], a[:, k], b[:, k]]
        s = util.fourier_series(t, self.T, C, hyperparam_k_list)
        s2 = s**2
        sin2 = (np.sin(trig_arg) / ls[0, k]) ** 2
        E = np.exp(-2*sin2)
        K = (s2 * s2.T) * E

        L = K * 4 * sin2 / ls[0, k]
        A0 = (s * s.T) * E * (s + s.T)

        _ , d_Diag = util.build_diagonal(t, hyperparam_list)

        dK_dl = np.trace(Q1.dot(L)) + np.trace(Q2.dot(L))
        dK_da0 = np.trace(Q1.dot(A0)) + np.trace(Q2.dot(A0))

        dK_dsigmas = np.empty((var_precision,1))
        for p in range(var_precision):
            dK_dsigmas[p] = np.trace(Q1.dot(2 * d_Diag[:, :, p])) + np.trace(Q2.dot(2*d_Diag[:,:,p]))

        dK_da = np.empty((n_fourier,1))
        dK_db = np.empty((n_fourier,1))

        for n in range(n_fourier):
            sin_term = np.sin(2*np.pi*(n+1)*t/self.T)
            cos_term = np.cos(2*np.pi*(n+1)*t/self.T)

            cos_s = cos_term * s.T
            sin_s = sin_term * s.T

            An = 2 * E * (s * s.T) * (cos_s + cos_s.T)
            Bn = 2 * E * (s * s.T) * (sin_s + sin_s.T)

            dK_da[n] = np.trace(Q1.dot(An)) + np.trace(Q2.dot(An))
            dK_db[n] = np.trace(Q1.dot(Bn)) + np.trace(Q2.dot(Bn))

        return dK_dl, dK_da0, dK_da, dK_db, dK_dsigmas
