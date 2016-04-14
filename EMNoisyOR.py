#!/usr/bin/env python
# EM learning algorithm for the Noisy-OR
# author: blue, 29/03/2016
# TODO add upwards translation to evaluation of meanPosterior

import numpy as np
import signal
import argparse

np.set_printoptions(precision=14, suppress=True, threshold=10000)

# Define parser
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--samplesfile', required=True, dest="sFile",
                    help='the npy file containing the samples')
parser.add_argument('-j', '--nhiddenvars', required=True, dest='nHiddenVars',
                    type=int, help='number of hidden variables')
parser.add_argument('-t', '--tparamsfile', required=True, dest="tFile",
                    help='the npz file containing the true parameters')
args = parser.parse_args()

def pseudoLogJoint(Pi, W, hiddenVarConfs, samples):
    """Takes the parameters and returns a matrix M[sample][hiddenVarConfs].
    Each element of the matrix is the pseudo-log-joint probablity \
                    B*log(p(hiddenVarConf, sample))"""

    # FIXME we already evaluate 1-W_dh*s_ch for the M-step
    # prods_dc = 1 - Wbar_dc = prod{h}{1-W_dh*s_ch}
    prods = np.prod(1 - np.einsum('ij,kj->ijk', W, hiddenVarConfs), axis=1)
    # logPriors_nc = sum{d}{y_nd*log(1/prods_dc - 1) + log(prods_dc)}
    logPriors = np.dot(samples, np.log(1/prods - 1)) + np.sum(np.log(prods), axis=0)
    # logHiddenVarProbs_c = sum{h}{hvc_ch}*log(Pi/(1-Pi))
    logHiddenVarProbs = np.sum(hiddenVarConfs, axis=1)*np.log(Pi/(1-Pi))
    # return pseudoLogJoints_cn
    return np.transpose(logHiddenVarProbs + logPriors)


def meanPosterior(g, pseudoLogJoints, samples):
    """Takes a (multidimensional) array g and returns its mean weighted
    over the posterior probabilities of each sample.
    The array is assumed to have the axis over which the mean is to
    be performed as last.

    An array with the same number of dimensions of g is returned, but now
    the last axis represents the mean of g relative to each different
    sample.
    
    The calculation performed is equivalent to np.dot(a, np.transpose(q))"""

    # sum{c}{g_ic*exp(pseudoLogJoints_cn)} / 
    #   (sum{c}{exp(pseudoLogJoints_cn)} + prod{d}{delta(y_nd))
    return np.dot(g, np.exp(pseudoLogJoints)) / \
            (np.sum(np.exp(pseudoLogJoints), axis=0) + \
                np.array(~np.any(samples, axis=1), dtype=int))

def pseudoLogL(pseudoLogJoints, samples):
    """Evaluate pseudoLogL = logL - N*H*log(1-Pi).
    pseudoLogL = sum{n}{log(prod{d}{delta(y_nd)} + \
            sum{c}{exp(pseudoLogJoints_cn)}}"""

    return np.sum(np.log(np.array(~np.any(samples, axis=1), dtype=int) + \
            np.sum(np.exp(pseudoLogJoints), axis=0)))


def debugPrint(pseudoLogJoints, Pi, W):
    #print "plg", pseudoLogJoints
    print "W", W
    #print "Pi", Pi


def signalHandler(signal, frame):
    global done
    print "Quitting on keyboard interrupt!"
    done = True


# Set problem data
nHiddenVars = args.nHiddenVars # Number of hidden variables assumed present
samples = np.load(args.sFile)
trueParams = np.load(args.tFile)
eps = 1e-13 # minimum value allowed for synaptic weights (max is 1-eps)

# Create array containing all possible hidden variables configurations
hiddenVarConfs = np.array([[0],[1]], dtype=int)
for i in range(nHiddenVars-1):
    hiddenVarConfs = np.vstack([ np.hstack(([x,x], [[0],[1]]))
                                 for x in hiddenVarConfs ])
# Remove the all-zero configuration. It is not required in the EM-algorithm.
hiddenVarConfs = np.delete(hiddenVarConfs, 0, 0)

# Initialise parameters to random values
Pi = np.random.rand()
W = np.random.rand(samples.shape[1], nHiddenVars) # W[dimSample][nHiddenVars]

# Initialise parameters to the ground-truth values
# Pi = trueParams["Pi"]
# W = trueParams["W"]

initW = W

# Evaluate true log-likelihood from true parameters (for consistency checks)
truePseudoLogL = pseudoLogL(pseudoLogJoint(trueParams["Pi"],
                                            trueParams["W"],
                                            hiddenVarConfs,
                                            samples),
                            samples)
signal.signal(signal.SIGINT, signalHandler)
done = False
counter = 0;
pseudoLogLs = ()
# first E-step: evaluate pseudo-log-joint probabilities
pseudoLogJoints = pseudoLogJoint(Pi, W, hiddenVarConfs, samples)
for i in range(100):
    # M-step
    Pi = np.sum(meanPosterior(np.sum(hiddenVarConfs, axis=1),
                              pseudoLogJoints,
                              samples)) / \
            (samples.shape[0]*nHiddenVars)

    # Wtilde_dhc = Prod{h'!=h}{1 - W_dj'*hiddenVarConfs_cj'}
    Ws = 1 - np.einsum('ij,kj->ijk',
                        W, hiddenVarConfs) # faster than * plus np.newaxis
    # TODO try with np.fromfunction instead?
    Wtilde = np.stack([ np.prod(np.delete(Ws, j, axis=1), axis=1) \
            for j in range(Ws.shape[1]) ], axis=1)
    denominators = 1 - Wtilde*Ws # faster than np.prod(Ws, axis=1)
    denominators = (1 - denominators)*denominators
    D = np.einsum('ijk,kj->ijk', Wtilde, hiddenVarConfs) / denominators
    # FIXME would it be faster not to save D and C and do all in one line?
    Dtilde = np.einsum('ijk,ki->ij',
                       meanPosterior(D, pseudoLogJoints, samples),
                       samples - 1)
    Ctilde = np.sum(meanPosterior(Wtilde*D, pseudoLogJoints, samples), axis=2)
    W = 1 + Dtilde/Ctilde
    np.clip(W, eps, 1-eps, out=W)

    # E-step: evaluate pseudo-log-joint probabilities
    pseudoLogJoints = pseudoLogJoint(Pi, W, hiddenVarConfs, samples)

    # Evaluate new likelihood and append it to the tuple
    pseudoLogLs += (pseudoLogL(pseudoLogJoints, samples),)

    # Print pseudoLogL to show progress
    counter += 1
    if counter % 10 == 0:
        debugPrint(pseudoLogJoints, Pi, W)
        print "pseudoLogL[" + str(counter) + "] = ", pseudoLogLs[-1]

# Evaluate errors
smallval = np.min(trueParams["W"])
bigval = np.max(trueParams["W"])
smalldiff = np.abs(W[W<0.5] - smallval)
bigdiff = np.abs(W[W>=0.5] - bigval) 
Werror = np.max(np.append(smalldiff, bigdiff))

filename = "l" + str(samples.shape[0])
np.savez(filename, Pi=Pi, W=W,
         pseudoLogLs=pseudoLogLs, truePseudoLogL=truePseudoLogL, initW=initW)
print "end Pi\n", Pi
print "end W (max error: " + str(Werror) + ")\n", W
print "results have been saved in " + filename + ".npz"
