#!/usr/bin/env python
# EM learning algorithm for the Noisy-OR
# author: blue, 29/03/2016
# TODO branch with weights shown in real-time
# TODO branch with version that allows multiple Pi's
# TODO branch with TV-EM
# TODO order bar matrices according to p(s_h' | y = W_dh). Delete sorting
#      routine in plotNoisyOR

import numpy as np
import signal
import argparse
from sys import exit

################ INITIAL SETUP #################
np.set_printoptions(precision=14, suppress=True, threshold=10000)

# Define option parser
parser = argparse.ArgumentParser()
parser.add_argument('-j', '--nhiddenvars', required=True, dest='nHiddenVars',
                    type=int, help='number of hidden variables')
parser.add_argument('-p', '--parFiles', dest='parFile',
                    help="""The body of the names of the files containing the
parameters. The samples file will be set to
'sPARFILE.npy' while the true parameters file will be set
to 'tPARFILE.npz'. This is a commodity option to save
typing. This option is overridden by the -s and -t options
if they are present.""")
parser.add_argument('-s', '--samplesfile', dest="sFile",
                    help='the npy file containing the samples')
parser.add_argument('-t', '--tparamsfile', dest="tFile",
                    help='the npz file containing the true parameters')
args = parser.parse_args()
if not args.parFile:
    if not (args.sFile and args.tFile):
        print """Please provide both samples and true prameters filenames.
The -p option can be used as a shorthand if filenames have the same body.
Please use the -h option for more information"""
        exit(1)


def pseudoLogJoint(Pi, W, hiddenVarConfs, samples, Ws):
    """Takes the parameters and returns a matrix M[hiddenVarConfs][sample].
Each element of the matrix is the pseudo-log-joint probablity \
B*log(p(hiddenVarConf, sample))"""

    # prods_dc = 1 - Wbar_dc = prod{h}{1-W_dh*s_ch}
    prods = np.prod(Ws, axis=1)
    # logPy_nc = sum{d}{y_nd*log(1/prods_dc - 1) + log(prods_dc)}
    logPy = np.dot(samples, np.log(1/prods - 1)) + \
                np.sum(np.log(prods), axis=0)
    # logPriors_c = sum{h}{hvc_ch}*log(Pi/(1-Pi))
    logPriors = np.sum(hiddenVarConfs, axis=1)*np.log(Pi/(1-Pi))
    # return pseudoLogJoints_cn
    return np.transpose(logPriors + logPy)


def meanPosterior(g, pseudoLogJoints, samples):
    """Takes a (multidimensional) array g and returns its mean weighted
    over the posterior probabilities of each sample.
    The array is assumed to have the axis over which the mean is to
    be performed as last.

    An array with the same number of dimensions of g is returned, but now
    the last axis represents the mean of g relative to each different
    sample.
    
    The calculation performed is equivalent to np.dot(a, np.transpose(q))
    wher q_cn are the posterior probabilities of each hidden variable
    configuration c give sample n"""

    # Evaluate constants B_n by which we can translate pseudoLogJoints
    # TODO check grafically whether 20 is a good magic number
    # (exp(20) is about the highest number we can evaluate precisely)
    B = 20 - np.max(pseudoLogJoints, axis=0)

    # sum{c}{g_ic*exp(pseudoLogJoints_cn + B)} /
    #   (sum{c}{exp(pseudoLogJoints_cn + B)} + prod{d}{delta(y_nd)*exp(B))
    return np.dot(g, np.exp(pseudoLogJoints + B)) / \
            (np.sum(np.exp(pseudoLogJoints + B), axis=0) + \
                deltaSamples*np.exp(B))


def logL(pseudoLogJoints, deltaSamples, nHiddenVars, Pi):
    """Evaluate log-likelihood logL
    logL = sum{n}{log(prod{d}{delta(y_nd)} + \
            sum{c}{exp(pseudoLogJoints_cn)}} + N*H*log(1-Pi)"""

    return np.sum(np.log(deltaSamples + \
           np.sum(np.exp(pseudoLogJoints), axis=0))) + \
           deltaSamples.size*nHiddenVars*np.log(1-Pi)


def debugPrint(pseudoLogJoints, Pi, W):
    print "plg", pseudoLogJoints
    print "W", W
    print "Pi", Pi


def signalHandler(signal, frame):
    global done
    print "Quitting on keyboard interrupt!"
    done = True


# Set problem data
# Number of hidden variables assumed present
nHiddenVars = args.nHiddenVars
# File containing the samples/datapoints
samples = np.load(args.sFile or ('s' + args.parFile + '.npy'))
# deltaSamples is a quantity useful for later calculations
# deltaSamples_n is 1 if samples_nd == 0 for each d, 0 otherwise
deltaSamples = np.array(~np.any(samples, axis=1), dtype=int)
# The ground-truth parameters
trueParams = np.load(args.tFile or ('t' + args.parFile + '.npz'))
# Minimum value allowed for synaptic weights (max is 1-eps)
eps = 1e-13

# Create array containing all possible hidden variables configurations
hiddenVarConfs = np.array([[0],[1]], dtype=int)
for i in range(nHiddenVars-1):
    hiddenVarConfs = np.vstack([ np.hstack(([x,x], [[0],[1]]))
                                 for x in hiddenVarConfs ])
# Do the same for the true number of hidden variables
trueHiddenVarConfs = np.array([[0],[1]], dtype=int)
for i in range(trueParams["W"].shape[1]-1):
    trueHiddenVarConfs = np.vstack([ np.hstack(([x,x], [[0],[1]]))
                                 for x in trueHiddenVarConfs ])

# Remove the all-zero configuration:
# it is not required in the EM-algorithm and causes 0/0 calculations
hiddenVarConfs = np.delete(hiddenVarConfs, 0, 0)
trueHiddenVarConfs = np.delete(trueHiddenVarConfs, 0, 0)

# Initialise parameters to random values
# TODO use smarter initial values for the parameters
Pi = np.random.rand()
W = np.random.rand(samples.shape[1], nHiddenVars) # W[dimSample][nHiddenVars]
initW = W # save initial values of the parameters for visualisation purposes

# Alternatively: initialise parameters to the ground-truth values
# Pi = trueParams["Pi"]
# W = trueParams["W"]

# Evaluate true log-likelihood from true parameters (for consistency checks)
trueWs = 1 - np.einsum('ij,kj->ijk', trueParams["W"], trueHiddenVarConfs)
trueLogL = logL(pseudoLogJoint(trueParams["Pi"],
                               trueParams["W"],
                               trueHiddenVarConfs,
                               samples,
                               trueWs),
                 deltaSamples,
                 trueHiddenVarConfs.shape[1],
                 trueParams["Pi"])


################ START LEARNING #################
# From now on Ctrl-C does not interrupt execution, just sets done = True
signal.signal(signal.SIGINT, signalHandler)
done = False
counter = 0;
logLs = () # The log-likelihoods evaluated at each step
for i in range(100):
# Alternatively:
# while not done:
    # Ws are values needed both in the E and M-step, we evaluate them once here
    # Ws_dhc = 1 - (W_dh * hiddenVarConfs_ch)
    Ws = 1 - np.einsum('ij,kj->ijk', W, hiddenVarConfs) 
    # E-step: evaluate pseudo-log-joint probabilities
    pseudoLogJoints = pseudoLogJoint(Pi, W, hiddenVarConfs, samples, Ws)

    # Evaluate new likelihood and append it to the tuple
    logLs += (logL(pseudoLogJoints, deltaSamples, nHiddenVars, Pi),)

    # M-step
    # Pi = sum{n}{<sum{h}{hiddenVarConfs_ch}>} / (N*H)
    Pi = np.sum(meanPosterior(np.sum(hiddenVarConfs, axis=1),
                              pseudoLogJoints,
                              samples)) / \
            (samples.shape[0]*nHiddenVars)

    # Wtilde_dhc = Prod{h'!=h}{1 - W_dj'*hiddenVarConfs_cj'}
    # (Wtilde = np.fromfunction is much slower than a loop over j + np.stack)
    Wtilde = np.stack([ np.prod(np.delete(Ws, j, axis=1), axis=1) \
                        for j in range(Ws.shape[1]) ], axis=1)
    denominators = 1 - Wtilde*Ws # faster than np.prod(Ws, axis=1) + newaxis
    denominators = (1 - denominators)*denominators
    D = np.einsum('ijk,kj->ijk', Wtilde, hiddenVarConfs) / denominators
    Dtilde = np.einsum('ijk,ki->ij',
                       meanPosterior(D, pseudoLogJoints, samples),
                       samples - 1)
    Ctilde = np.sum(meanPosterior(Wtilde*D, pseudoLogJoints, samples), axis=2)
    W = 1 + Dtilde/Ctilde
    np.clip(W, eps, 1-eps, out=W)

    if counter % 10 == 0:
        # Print logL to show progress
        print "logL[" + str(counter) + "] = ", logLs[-1]
    counter += 1


################ WRAP-UP OPERATIONS #################
# Evaluate max difference with true values of a bars test
smallval = np.min(trueParams["W"])
bigval = np.max(trueParams["W"])
smalldiff = np.abs(W[W<0.5] - smallval)
bigdiff = np.abs(W[W>=0.5] - bigval) 
Werror = np.max(np.append(smalldiff, bigdiff))

# Save results to file and print out last parameter values
filename = "l" + str(samples.shape[0])
np.savez(filename, Pi=Pi, W=W,
         logLs=logLs, trueLogL=trueLogL, initW=initW)
print "end Pi\n", Pi
print "end W (max error: " + str(Werror) + ")\n", W
print "results have been saved in " + filename + ".npz"
