#!/usr/bin/env python
# generate samples from a Noisy-OR mixture
# author: blue, 29/03/2016

import numpy as np
from scipy.stats import bernoulli
from math import sqrt
import argparse

# Defiine parser
parser = argparse.ArgumentParser()
parser.add_argument('-j', '--nHiddenVars', default=8, dest='nHiddenVars',
                    type=int, help='number of hidden variables')
parser.add_argument('-n', '--nsamples', default=500, dest='nSamples', type=int,
                    help='number of samples to generate')
parser.add_argument('-d', '--dimSample', default=16, dest='dimSample', type=int,
                    help='linear dimension of each generated sample, i.e.\
                          size of output array')
args = parser.parse_args()

# Define parameters
nSamples = args.nSamples
nHiddenVars = args.nHiddenVars
dimSample = args.dimSample

# Each W[h,:] is seen as a sqrt(dimSample)xsqrt(dimSample) matrix
# e.g. matrices will be 5x5 if dimSample==25
# Each of this matrices has one vertical or horizontal bar
# The number of matrices should be equal to the number of hidden variables
W = np.ones((dimSample, nHiddenVars))*0.1
dimMatrix = int(sqrt(dimSample))
nBars = min(nHiddenVars, 2*dimMatrix)
value = 0.8
# Paint vertical bars
for c in range(nBars/2):
    W[[ i*dimMatrix + c for i in range(dimMatrix) ], c ] = value
# Paint horizontal bars
for c in range(nBars/2):
    W[[ i + c*dimMatrix for i in range(dimMatrix) ], c + nBars/2 ] = value

# We want an average of 2 bars/sample
Pi = 2./nHiddenVars

samples = []
for i in range(nSamples):
    # Generate hidden variables array s
    s = bernoulli.rvs(Pi, size=nHiddenVars)
    # Evaluate array of bernoulli probabilities for the samples y
    yProb = 1 - np.prod(1 - W*s, axis=1)
    # produce a sample and put it in samples
    samples.append([ bernoulli.rvs(yProb[d]) for d in range(dimSample) ])

samplesArray = np.array(samples, int)
np.set_printoptions(threshold=nHiddenVars-1)
print "samples\n", samplesArray
print "\nPi", Pi
print "\nW\n", W
np.save("s" + str(nSamples), samplesArray)
np.savez("t" + str(nSamples), Pi=Pi, W=W)
print "parameters were saved in file t" + str(nSamples) + ".npz"
print "samples were saved in file s" + str(nSamples) + ".npy"
