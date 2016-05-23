#!/usr/bin/env python
# Plot log-likelihood and compare true parameters with learned parameters
# author: blue, 29/03/2016

import numpy as np
import matplotlib.pyplot as plt
import argparse

# Define parser
parser = argparse.ArgumentParser()
parser.add_argument('-l', '--learned', dest="learnedp",
                    help='the npz file containing the learned parameters')
parser.add_argument('-d', '--dps', dest="dps", default="",
                    help='the npy file containing the data-points')
args = parser.parse_args()

def clear_axes(plot):
    axes = plot.get_axes()
    axes.set_xticklabels([])
    axes.set_xticks([])
    axes.set_yticklabels([])
    axes.set_yticks([])

# PRINT LOG-LIKELIHOOD
# Load log-likelihood and learned parameters
lp = np.load(args.learnedp or ('L' + args.parFile + '.npz'))
# Plot log-likelihood and true log-likelihood
plt.figure()
plt.plot(range(lp["logLs"].size),
         lp["logLs"])
plt.title("Learned log-likelihood")
plt.xlabel("iterations")
plt.ylabel("log-likelihood")


# PRINT INITIAL AND LEARNED WEIGHTS
# Build grid of heat-maps to compare true and learned parameters
nMatrices = lp["W"].shape[1]
# matrices[0] is the initial matrices, matrices[1] is the learned ones,
# matrices[2] the ground-truth matrices
matrices = [ np.transpose(lp["initW"]).reshape(lp["W"].shape[1],
                                               27,
                                               10), 
             np.transpose(lp["W"]).reshape(lp["W"].shape[1],
                                           27,
                                           10) ]
titles = ("Initial weights", "Learned weights", "True weights")
plt.figure(figsize=(plt.rcParams['figure.figsize'][0],
                    plt.rcParams['figure.figsize'][1]*2))
for nMat in range(nMatrices):
    for i in range(2):
        if matrices[i].shape[0] > nMat:
            plt.subplot(nMatrices, 2, nMat*2+i+1)
            plot = plt.imshow(matrices[i][nMat], cmap="Greys", aspect='equal',
                              vmin=0, vmax=1, interpolation='none')
            clear_axes(plot)
            if nMat == 0:
                plt.title(titles[i])
        
# PRINT PI'S
plt.figure()
plot = plt.imshow(np.atleast_2d(lp["Pi"]), interpolation='none',
                  vmin=0., vmax=.5)
clear_axes(plot)
plt.title("Subpopulations' probabilities")
clear_axes(plot)
plt.colorbar()
plt.show()
