#!/usr/bin/env python
# Infer posterior probabilities for the specified hidden configurations
# author: Enrico Guiraud, 04/07/2016
# TODO document code better

import numpy as np
import emutils as em
import argparse
from math import sqrt

# Define parser
parser = argparse.ArgumentParser()
parser.add_argument('-l', '--learned', dest='learnedp', required=True,
                    help='the npz file containing the learned parameters')
parser.add_argument('-d', '--data', dest='data', required=True,
                    help='data-points on which to perform inference')
parser.add_argument('-g', '--plots', dest='plot', action='store_true',
                    help='display inferred hidden states graphically, i.e. \
                    display their weights')
parser.add_argument('-s', '--states', dest='Sfile',
                    help='specify a numpy file containing the hidden variables \
                    configurations for which the script should do inference. \
                    The default is performing inference on all single-cause \
                    states')
args = parser.parse_args()

# Load learned parameters and data-points
lp = np.load(args.learnedp)
Y = np.load(args.data)
# Extract parameters from the data
D, H = lp["W"].shape
N = Y.shape[0]
# Set hidden variables configurations for which to evaluate posteriors
if args.Sfile:
    S = np.load(args.Sfile)
else:
    S = np.eye(H)

# Check everything is alright
assert D == Y.shape[1], "mismatch in dimensions of W and Y (respectively " \
                        + str(lp["W"].shape) + " and " + str(Y.shape) + ")"
assert H == S.shape[1], "mismatch in dimensions of W and S (respectively " \
                        + str(lp["W"].shape) + " and " + str(S.shape) + ")"

# Evaluate posteriors for each data-point and each hidden configuration
# psgy has shape (len(S), N)
psgy = em.posterior(lp["Pi"], lp["W"], S, Y)

# For each data-point, order the hidden configurations by decreasing posterior
sortedInd = np.argsort(psgy, axis=0)[::-1, :]
sortedPsgy = psgy[sortedInd.flatten(), range(psgy.shape[1])*psgy.shape[0]]
bestConfs = np.empty((N, S.shape[0], H))
for n in range(N):
    bestConfs[n] = S[sortedInd[:,n]]
np.savez("inference", S=bestConfs, psgy=sortedPsgy)
print "results saved in file inference.npz"

# plot things
if args.plot == True:
    import matplotlib.pyplot as plt
    W = lp["W"]
    dim = int(sqrt(Y.shape[1]))
    zeroW = np.zeros((dim,dim))
    for n in range(N):
        nActiveS = int(bestConfs[n,0].sum())
        # Plot the corresponding weights of each MAP configuration
        plt.figure()
        plt.subplot(1, max(nActiveS + 1, 2), 1)
        plt.imshow(Y[n].reshape(dim,dim),
                   cmap="Greys", interpolation="none")
        if bestConfs[n,0].sum() == 0:
            # no active components
            plt.subplot(1, max(nActiveS + 1, 2), 2)
            plt.imshow(zeroW, cmap="Greys", interpolation="none")
        else:
            activeS = np.nonzero(bestConfs[n,0])[0]
            for i in range(nActiveS):
                plt.subplot(1, max(nActiveS + 1, 2), i + 2)
                plt.imshow(W[:,activeS[i]].reshape(dim,dim),
                           cmap="Greys", interpolation="none")
    plt.show()
