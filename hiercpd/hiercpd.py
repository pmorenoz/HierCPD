# Copyright (c) 2018 Pablo Moreno-Munoz
# Universidad Carlos III de Madrid

import numpy as np
import random
import matplotlib.pyplot as plt

class HierCPD():
    def __init__(self, Z, hyp_lambda=None, name='HierCPD'):

        self.Z = Z
        self.T = Z.shape[0]
        self.Znan = np.isnan(self.Z)
        self.K = np.int(np.nanmax(self.Z))
        self.Zbin = self.onehot()
        self.maxes = np.empty((self.T+1,1))

        # Run length initialization
        self.R = np.zeros((self.T+1, self.T+1))
        self.R[0,0] = 1

        # Hyperparameters initialization
        if hyp_lambda is not None:
            self.hyp_lambda = hyp_lambda
        else:
            pass
        self.alpha0 = np.ones((1,self.K))
        self.alphaT = self.alpha0

        # Maxes initialization
        self.maxes[0] = 0

    def detection(self):
        # Detector
        for t in range(self.T):
            if self.Znan[t]:
                pred = 1.
            else:
                pred = self.alphaT[:, self.Zbin[t,:]==1] / np.sum(self.alphaT,1)[:,None]

            H = self.constant_hazard(np.arange(t+1)+1)
            self.R[1:t+2, t+1, None] = self.R[0:t+1, t, None] * pred * (1-H)
            self.R[0,t+1] = np.sum(self.R[0:t+1, t] * pred * H)
            self.R[:,t+1] = self.R[:,t+1] / np.sum(self.R[:, t+1])

            if sum(self.Zbin[t,:] > 1):
                alphaT0 = np.vstack(( self.alpha0, self.alphaT))
            else:
                alphaT0 = np.vstack((self.alpha0, self.alphaT + np.tile(self.Zbin[t,:].T, (self.alphaT.shape[0] ,1)) ))

            self.alphaT = alphaT0
            self.maxes[t+1] = np.where(self.R[1:,t+1] == np.max(self.R[1:,t+1]))

        self.maxes[1:] = self.maxes[1:] + 1
        #self.maxes[self.T] = np.where(self.R[:, self.T] == np.max(self.R[:, self.T]))

    def constant_hazard(self, R):
        # Probability of CP per unit of time
        H = (1./self.hyp_lambda) * np.ones((R.shape[0],1))
        return H


    def onehot(self):
        # One-Hot Encoding of Categorical Data
        Z_onehot = np.zeros((self.T, self.K))
        for k in range(self.K):
            Z_onehot[:, k, None] = (self.Z==k+1).astype(np.int)

        return Z_onehot

    def plot_detection(self):
        plt.figure(figsize=(6,8))
        ax1 = plt.subplot(211)
        plt.imshow(self.Zbin.T, cmap='gray_r', aspect='auto')
        ax2 = plt.subplot(212, sharex=ax1)
        plt.imshow(-np.log(self.R), cmap='gray', aspect='auto')
        plt.plot(self.maxes, color='red', linewidth=2.0)
        plt.show()


