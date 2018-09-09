# Copyright (c) 2018 Pablo Moreno-Munoz
# Universidad Carlos III de Madrid and eB2 (eb2.tech)

import numpy as np
import matplotlib.pyplot as plt

from GPy import kern
from GPy.util import linalg
from scipy.stats import multivariate_normal
from scipy.stats import bernoulli

def build_covariance(t, K, hyperparams):

    ls = hyperparams[0]
    a0 = hyperparams[1]
    a = hyperparams[2]
    b = hyperparams[3]

    C,_ = a.shape # number of Fourier coefficients

    T,_ = t.shape
    S = np.empty((T,T,K))
    L = np.empty((T,T,K))
    Si = np.empty((T,T,K))

    Diag, _ = build_diagonal(t, hyperparams)
    for k in range(K):
        # falta meter el término periódic
        hyperparam_k_list = [ls[0,k], a0[0,k], a[:,k], b[:,k]]
        per_term = fourier_series(t, T, C, hyperparam_k_list)
        s = per_term**2
        per_S = s * s.T
        E = periodic_exponential(t, T, hyperparam_k_list)
        S[:,:,k] = per_S * E
        S[:,:,k] += Diag
        L[:,:,k] = linalg.jitchol(S[:,:,k])
        Si[:,:,k], _ = linalg.dpotri(np.asfortranarray(L[:,:,k]))

        # quitar esto:
        #S[:,:,k] = np.eye(T, T)
        #Si[:,:,k] = np.eye(T, T)
    return S, L, Si

def build_diagonal(t, hyperparams):
    sigmas = hyperparams[4]
    var_precision = sigmas.shape[0]
    T = t.shape[0]

    var_vector = np.ones((T,1))
    per_length = np.int(T/var_precision)
    for p in range(var_precision):
        var_vector[per_length*p:per_length*(p+1)] = sigmas[p]*var_vector[per_length*p:per_length*(p+1)]

    var_vector = var_vector.flatten()
    Diag = np.diag(var_vector**2)

    # Matrix of derivatives: (T,T,var_precision)
    d_Diag = np.empty((T,T,var_precision))
    for p in range(var_precision):
        d_Diag_aux = np.zeros((T,T))
        d_Diag_aux[per_length * p:per_length * (p + 1), per_length * p:per_length * (p + 1)] = np.diag(var_vector[per_length*p:per_length*(p+1)])
        d_Diag[:,:,p] = d_Diag_aux
    return Diag, d_Diag

def heterogeneous_pdf(Y,params):
    N,T,_ = Y.shape
    Yreal = Y[:,:,0]
    Ybin = Y[:,:,1]

    notnans = np.invert(np.isnan(Yreal))
    f = params[0]
    S = params[1]
    #Si = params[2]
    mu = params[3]

    n_pdf = np.empty((1,N))
    b_pdf = np.empty((1, N))
    # make this faster!!
    for i in range(N):
        n_pdf[0,i] = multivariate_normal(mean=f[notnans[i,:]].flatten(), cov=S[np.ix_(notnans[i,:], notnans[i,:])]).pdf(Yreal[i,notnans[i,:]])
        b_pdf[0,i] = np.prod(bernoulli(mu[notnans[i,:]]).pmf(Ybin[i,notnans[i,:]]))

    het_pdf = n_pdf * b_pdf
    return het_pdf

def heterogeneous_logpdf(Y,params):
    # Note that this logpdf is calculated over the filled observations (see previous expectations)
    N,T,_ = Y.shape
    Yreal = Y[:,:,0]
    Ybin = Y[:,:,1]

    f = params[0]
    S = params[1]
    #Si = params[2]
    mu = params[3]

    n_logpdf = np.empty((1,N))
    b_logpdf = np.empty((1, N))
    # make this faster!!
    for i in range(N):
        n_logpdf[0,i] = multivariate_normal(mean=f.flatten(), cov=S).logpdf(Yreal[i,:])
        b_logpdf[0,i] = np.prod(bernoulli(mu).logpmf(Ybin[i,:]))

    het_logpdf = n_logpdf * b_logpdf
    return het_logpdf


def fourier_series(t, T, C, hyperparams):
    a0 = hyperparams[1]
    a = hyperparams[2]
    b = hyperparams[3]

    x = np.tile(t, (1,C))

    s = 0.5 * a0
    cos_args = 2 * np.pi * np.arange(1, C+1) / T
    sin_args = 2 * np.pi * np.arange(1, C+1) / T
    cos_term = np.cos(np.tile(cos_args[np.newaxis,:], (T, 1)) * x)
    sin_term = np.sin(np.tile(sin_args[np.newaxis, :], (T, 1)) * x)

    s += np.sum(a * cos_term, 1) + np.sum(b * sin_term, 1)
    return s[:,np.newaxis]

def periodic_exponential(t, T, hyperparams):
    ls = hyperparams[0]
    trig_arg = np.pi * (t - t.T) / T
    sin2 = (np.sin(trig_arg) / ls) ** 2
    E = np.exp(-2 * sin2)
    return E

