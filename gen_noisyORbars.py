#!/usr/bin/env python
# generate data-points from a Noisy-OR generative model
# author: Enrico Guiraud, 26/05/2016

# This script generates N data-points of dimension D with a noisy-OR
# generative model. The priors are taken all equal to 2/H, where H is the
# number of hidden variables, and the weight matrices are single horizontal
# and vertical bars. The data-points will therefore consist of a superposition
# of an average of 2 bars, with noise.

import numpy as np
from scipy.stats import bernoulli
from math import sqrt
import argparse
import emutils as em

# Define parser
parser = argparse.ArgumentParser()
parser.add_argument('-H', '--nHiddenVars', required=True, dest='H',
                    type=int, help='number of hidden variables')
parser.add_argument('-N', '--ndps', required=True, dest='N', type=int,
                    help='number of data-points to generate')
parser.add_argument('-D', '--dimdp', default=0, dest='D', type=int,
                    help='linear dimension of each generated data-point, i.e.\
                          size of output array. Default value is (H/2)^2')
args = parser.parse_args()

# Define parameters
N = args.N
H = args.H
D = args.D
if not D:
    D = (H / 2) ** 2

# Each W[:,h] is seen as a sqrt(D)xsqrt(D) matrix
# e.g. matrices will be 5x5 if D==25
# Each of this matrices has one vertical or horizontal bar
# The number of matrices should be equal to the number of hidden variables
dimMatrix = int(sqrt(D))
nBars = min(H, 2 * dimMatrix)
bgval, barsval = (0.1, 0.8)
W = np.ones((D, H)) * bgval
# Paint vertical bars
for c in range(nBars / 2):
    W[[i * dimMatrix + c for i in range(dimMatrix)], c] = barsval
# Paint horizontal bars
for c in range(nBars / 2):
    W[[i + c * dimMatrix for i in range(dimMatrix)], c + nBars / 2] = barsval

# We want an average of 2 bars/data-point, plus noise
Pi = np.array([ 2./H ]*H) \
     + np.random.uniform(low=max(-0.2, -1./H), high=0.2, size=H)
# Normalise Pi to have an average of two bars per figure
Pi = Pi / Pi.sum() * 2

Y = []
for i in range(N):
    # Generate hidden variables array s
    s = np.array([ bernoulli.rvs(p) for p in Pi ])
    # Evaluate array of bernoulli probabilities for the data-points y
    yProb = 1 - np.prod(1 - W * s, axis=1)
    # produce a data-point and put it in data-points
    Y.append([bernoulli.rvs(yProb[d]) for d in range(D)])
Y = np.array(Y, int)

# Evaluate true log-likelihood from true parameters (for consistency checks)
S = em.genHiddenVarStates(H)
deltaY = np.array(~np.any(Y, axis=1), dtype=int)
trueLogL = em.logL(em.pseudoLogJoint(Pi, W, S, Y),
                   deltaY, H, Pi)

np.set_printoptions(threshold=H - 1)
print "data-point size", len(Y[0])
print "Pi", Pi
print "background, bars = (" + str(bgval) + ", " + str(barsval) + ")"
print "true logL =", trueLogL
np.save("N" + str(N), Y)
np.savez("T" + str(N), Pi=Pi, W=W, trueLogL=trueLogL)
print "parameters were saved in file T" + str(N) + ".npz"
print "data-points were saved in file N" + str(N) + ".npy"
