#!/usr/bin/env python
# generate data-points from given noisyOR parameters or from ground-truth shapes
# author: Enrico Guiraud, 25/07/2016

# TODO explain what the script does, write example usage

import numpy as np
import argparse
import emutils as em
import sys
from math import sqrt

# Define parser
parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-g', '--groundTruth', dest='gt',
                   help='npy file containing weights W and Pi')
group.add_argument('-s', '--shapes', dest='shapes',
                   help='npy file containing the shapes to combine via noisyOR')
parser.add_argument('-N', '--nDatapoints', dest='N', type=int, required=True,
                    help='number of data-points to generate')
parser.add_argument('-l', '--logL', dest='logL', action="store_true",
                    help='also evaluate the true log-likelihood of the \
                    generated data-points')
args = parser.parse_args()

bgval, fgval = (0.1, 0.9)
# Retrieve parameters
N = args.N
if args.shapes:
    shapes = np.load(args.shapes)
    W = shapes.T.clip(bgval, fgval)
    Pi = 2. / shapes.shape[0]
    print "Pi has been set to a uniform value of", Pi
else:
    p = np.load(args.gt)
    if not p.has_key("W"):
        print "Error: ground truth parameter file must contain a numpy array \
        named \"W\""
        sys.exit(1)
    W = p["W"]
    if not p.has_key("Pi"):
        Pi = 2. / W.shape[1]
        print "Pi has been set to a uniform value of", Pi
    else:
        assert Pi.shape == (1,) or Pi.shape == (W.shape[1],), \
               "Error: Pi parameter has an invalid shape"
        Pi = p["Pi"]
D, H = W.shape
# Each W[:,h] is seen as a sqrt(D)xsqrt(D) matrix
# e.g. matrices will be 5x5 if D==25
dimMatrix = int(sqrt(D))

Y = []
for i in range(N):
    # Generate hidden variables array s
    s = np.random.rand(H) < Pi
    # Evaluate array of bernoulli probabilities for the data-points y
    yProb = 1 - np.prod(1 - W * s, axis=1)
    # produce a data-point and put it in data-points
    Y.append(np.random.rand(D) < yProb)
Y = np.array(Y, int)

if args.logL:
    # Evaluate true log-likelihood from true parameters (for consistency checks)
    S = em.genHiddenVarStates(H)
    deltaY = np.array(~np.any(Y, axis=1), dtype=int)
    trueLogL = em.logL(em.pseudoLogJoint(Pi, W, S[None, ...], Y),
                       deltaY, H, Pi)
    print "true log-likelihood is", trueLogL

np.save("N" + str(N), Y)
print "data-points were saved in file N" + str(N) + ".npy"
if args.logL:
    np.savez("T" + str(N), Pi=Pi, W=W, trueLogL=trueLogL)
else:
    np.savez("T" + str(N), Pi=Pi, W=W)
print "parameters were saved in file T" + str(N) + ".npz"
