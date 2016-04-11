#!/usr/bin/env python
# Plot log-likelihood and compare true parameters with learned parameters
# author: blue, 29/03/2016

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import argparse

# Define parser
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--truth', dest="truep", required=True,
                    help='the npz file containing the true parameters')
parser.add_argument('-l', '--learned', dest="learnedp", required=True,
                    help='the npz file containing the learned parameters')
parser.add_argument('-s', '--samples', dest="samples", required=True,
                    help='the npy file containing the samples')
args = parser.parse_args()

# Load true parameters
tp = np.load(args.truep)
# Load log-likelihood and learned parameters
lp = np.load(args.learnedp)
# Samples
samples = np.load(args.samples)
# Plot log-likelihood and true log-likelihood
plt.figure()
plt.plot(range(lp["logL"].size), lp["logL"],
         range(lp["logL"].size), lp["trueLogL"].repeat(lp["logL"].size))
plt.title("True vs learned log-likelihood")
# Plot the first twelve samples
dimMatrix = sqrt(samples.shape[1])
samples = samples[:12]
plt.figure()
for nPlot in range(12):
    plt.subplot(4, 3, nPlot+1)
    plt.pcolor(samples[nPlot].reshape(dimMatrix, dimMatrix),
                 cmap="Greys")
    if nPlot == 1:
            plt.title('First 12 samples')

# Build grid of heat-maps to compare true and learned parameters
nMatrix = tp["W"].shape[1]
plt.figure()
for nPlot in range(nMatrix):
        plt.subplot(nMatrix, 3, nPlot*3+1)
        plt.pcolor(tp["W"][:,nPlot].reshape(dimMatrix, dimMatrix),
                     cmap="Greys")
        if nPlot == 0:
                plt.title('True weights')
        plt.subplot(nMatrix, 3, nPlot*3+2)
        plt.pcolor(lp["W"][:,nPlot].reshape(dimMatrix, dimMatrix),
                     cmap="Greys")
        if nPlot == 0:
                plt.title('Learned weights')
        plt.subplot(nMatrix, 3, nPlot*3+3)
        plt.pcolor(lp["initW"][:,nPlot].reshape(dimMatrix, dimMatrix),
                     cmap="Greys")
        if nPlot == 0:
                plt.title('Initial weights')
        
plt.figure()
plt.pcolor(np.vstack((np.sort(lp["hiddenVarWeights"]),
                      np.sort(tp["hiddenVarWeights"]))),
           vmin=0., vmax=1.)
plt.colorbar()
plt.title("Subpopulations' probabilities\n(true on top of learned)")

plt.show()
