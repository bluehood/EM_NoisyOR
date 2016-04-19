#!/usr/bin/env python
# Plot log-likelihood and compare true parameters with learned parameters
# author: blue, 29/03/2016
# TODO make it possible to have different numbers of true and learned matrices
# TODO sort learned matrices to show besides the corresponding true matrix

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

def clear_axes(plot):
    axes = plot.get_axes()
    axes.set_xticklabels([])
    axes.set_xticks([])
    axes.set_yticklabels([])
    axes.set_yticks([])

# Load true parameters
tp = np.load(args.truep)
# Load log-likelihood and learned parameters
lp = np.load(args.learnedp)
# Samples
samples = np.load(args.samples)
# Plot log-likelihood and true log-likelihood
plt.figure()
plt.plot(range(lp["logLs"].size),
         lp["logLs"],
         range(lp["logLs"].size),
         lp["trueLogL"].repeat(lp["logLs"].size))
plt.title("True vs learned log-likelihood")
plt.xlabel("iterations")
plt.ylabel("log-likelihood")
# Plot the first twelve samples
dimMatrix = sqrt(samples.shape[1])
samples = samples[:12]

plt.figure()
for nPlot in range(12):
    plt.subplot(4, 3, nPlot+1)
    plot = plt.pcolor(samples[nPlot].reshape(dimMatrix, dimMatrix),
                 cmap="Greys")
    clear_axes(plot)
    if nPlot == 1:
            plt.title('First 12 samples')

# Build grid of heat-maps to compare true and learned parameters
nMatrices = tp["W"].shape[1]
tMats = np.transpose(tp["W"]).reshape(nMatrices,dimMatrix,dimMatrix)
lMats = np.transpose(lp["W"]).reshape(nMatrices,dimMatrix,dimMatrix)
iMats = np.transpose(lp["initW"]).reshape(nMatrices,dimMatrix,dimMatrix)
plt.figure()
for nMat in range(nMatrices):
        plt.subplot(nMatrices, 3, nMat*3+1)
        plot = plt.pcolor(iMats[nMat], cmap="Greys", vmin=0, vmax=1)
        clear_axes(plot)
        if nMat == 0:
                plt.title('Initial weights')
        plt.subplot(nMatrices, 3, nMat*3+2)
        plot = plt.pcolor(lMats[nMat], cmap="Greys", vmin=0, vmax=1)
        clear_axes(plot)
        if nMat == 0:
                plt.title('Learned weights')
        plt.subplot(nMatrices, 3, nMat*3+3)
        plot = plt.pcolor(tMats[nMat], cmap="Greys", vmin=0, vmax=1)
        clear_axes(plot)
        if nMat == 0:
                plt.title('True weights')
        
plt.figure()
plot = plt.pcolor(np.vstack((lp["Pi"],tp["Pi"])),
           vmin=0., vmax=.5)
plt.colorbar()
axes = plot.get_axes()
axes.set_xticks([])
axes.set_xticklabels([])
axes.set_yticks([0.5, 1.5])
axes.set_yticklabels(["learned", "ground-truth"])
plt.title("Subpopulations' probabilities")

plt.show()
