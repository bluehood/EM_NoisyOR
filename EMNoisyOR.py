#!/usr/bin/env python
# EM learning algorithm for the Noisy-OR
# author: blue, 29/03/2016
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
parser.add_argument('-H', '--nhiddenvars', required=True, dest='nHiddenVars',
                    type=int, help='number of hidden variables')
parser.add_argument('-d', '--dpsfile', required=True, dest="dataFile",
                    help='the npy file containing the data-points')
parser.add_argument('-t', '--tparamsfile', dest="tFile", required=True,
                    help='the npz file containing the true parameters')
args = parser.parse_args()


def evaluateWtilde(Ws):
    # Wtilde_dhc = Prod{h'!=h}{1 - W_dj'*hiddenVarConfs_cj'}
    # These three lines work by multiplying cumulative products of Ws in both
    # directions (from beginning to end of each row and from end to beginning)
    # in a smart way. If Ws = array([a,b,c]), the contents of ret would be,
    # for each of the lines, ret == [1,1,1], then ret == [1, a, ab], and
    # finally ret == [1, a, ab]*[cb, c, 1] == [cb, ac, ab]
    ret = np.ones_like(Ws)
    np.cumprod(Ws[:, :-1], axis=1, out=ret[:, 1:])
    ret[:, :-1] *= np.cumprod(Ws[:, :0:-1], axis=1)[:, ::-1]
    return ret


def pseudoLogJoint(Pi, W, hiddenVarConfs, dps, Ws):
    """Takes the parameters and returns a matrix M[hiddenVarConfs][dp].
Each element of the matrix is the pseudo-log-joint probablity \
B*log(p(hiddenVarConf, dp))"""

    # prods_dc = 1 - Wbar_dc = prod{h}{1-W_dh*s_ch}
    prods = np.prod(Ws, axis=1)
    # logPy_nc = sum{d}{y_nd*log(1/prods_dc - 1) + log(prods_dc)}
    logPy = np.dot(dps, np.log(1/prods - 1)) + \
                np.sum(np.log(prods), axis=0)
    # logPriors_c = sum{h}{hvc_ch}*log(Pi/(1-Pi))
    logPriors = np.sum(hiddenVarConfs, axis=1)*np.log(Pi/(1-Pi))
    # return pseudoLogJoints_cn
    return np.transpose(logPriors + logPy)


def meanPosterior(g, pseudoLogJoints, dps):
    """Takes a (multidimensional) array g and returns its mean weighted
    over the posterior probabilities of each data-point.
    The array is assumed to have the axis over which the mean is to
    be performed as last.

    An array with the same number of dimensions of g is returned, but now
    the last axis represents the mean of g relative to each different
    data-point.
    
    The calculation performed is equivalent to np.dot(a, np.transpose(q))
    wher q_cn are the posterior probabilities of each hidden variable
    configuration c give data-point n"""

    # Evaluate constants B_n by which we can translate pseudoLogJoints
    B = 200 - np.max(pseudoLogJoints, axis=0)

    # sum{c}{g_ic*exp(pseudoLogJoints_cn + B)} /
    #   (sum{c}{exp(pseudoLogJoints_cn + B)} + prod{d}{delta(y_nd)*exp(B))
    return np.dot(g, np.exp(pseudoLogJoints + B)) / \
            (np.sum(np.exp(pseudoLogJoints + B), axis=0) + \
                deltaDps*np.exp(B))


def logL(pseudoLogJoints, deltaDps, nHiddenVars, Pi):
    """Evaluate log-likelihood logL
    logL = sum{n}{log(prod{d}{delta(y_nd)} + \
            sum{c}{exp(pseudoLogJoints_cn)}} + N*H*log(1-Pi)"""

    return np.sum(np.log(deltaDps + \
           np.sum(np.exp(pseudoLogJoints), axis=0))) + \
           deltaDps.size*nHiddenVars*np.log(1-Pi)


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
# File containing the data-points/data-points
dps = np.load(args.dataFile)
# deltaDps is a quantity useful for later calculations
# deltaDps_n is 1 if dps_nd == 0 for each d, 0 otherwise
deltaDps = np.array(~np.any(dps, axis=1), dtype=int)
# The ground-truth parameters
trueParams = np.load(args.tFile or ('t' + args.parFile + '.npz'))
# Minimum value allowed for synaptic weights (max is 1-eps)
eps = 1e-2

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

# Initialise parameters
Pi = 1./nHiddenVars
W = np.random.rand(dps.shape[1], nHiddenVars)
np.clip(W, eps, 1-eps, out=W)
# Alternatively: initialise parameters to the ground-truth values
# Pi = trueParams["Pi"]
# W = trueParams["W"]

initW = np.copy(W) # save initial values of the parameters

# Evaluate true log-likelihood from true parameters (for consistency checks)
trueWs = 1 - np.einsum('ij,kj->ijk', trueParams["W"], trueHiddenVarConfs)
trueLogL = logL(pseudoLogJoint(trueParams["Pi"],
                               trueParams["W"],
                               trueHiddenVarConfs,
                               dps,
                               trueWs),
                 deltaDps,
                 trueHiddenVarConfs.shape[1],
                 trueParams["Pi"])
print "true logL =", trueLogL


################ START LEARNING #################
# From now on Ctrl-C does not interrupt execution, just sets done = True
signal.signal(signal.SIGINT, signalHandler)
done = False
counter = 0;
logLs = () # The log-likelihoods evaluated at each step
for i in range(100):
    if done:
        break
# Alternatively:
# while not done:
    # Ws are values needed both in the E and M-step, we evaluate them once here
    # Ws_dhc = 1 - (W_dh * hiddenVarConfs_ch)
    Ws = 1 - np.einsum('ij,kj->ijk', W, hiddenVarConfs) 
    # E-step: evaluate pseudo-log-joint probabilities
    pseudoLogJoints = pseudoLogJoint(Pi, W, hiddenVarConfs, dps, Ws)

    # Evaluate new likelihood and append it to the tuple
    logLs += (logL(pseudoLogJoints, deltaDps, nHiddenVars, Pi),)

    # M-step
    # Pi = sum{n}{<sum{h}{hiddenVarConfs_ch}>} / (N*H)
    Pi = np.sum(meanPosterior(np.sum(hiddenVarConfs, axis=1),
                              pseudoLogJoints,
                              dps)) / \
            (dps.shape[0]*nHiddenVars)

    Wtilde = evaluateWtilde(Ws)
    denominators = 1 - Wtilde*Ws # faster than np.prod(Ws, axis=1) + newaxis
    denominators = (1 - denominators)*denominators
    D = np.einsum('ijk,kj->ijk', Wtilde, hiddenVarConfs) / denominators
    Dtilde = np.einsum('ijk,ki->ij',
                       meanPosterior(D, pseudoLogJoints, dps),
                       dps - 1)
    Ctilde = np.sum(meanPosterior(Wtilde*D, pseudoLogJoints, dps), axis=2)
    W = 1 + Dtilde/Ctilde
    np.clip(W, eps, 1-eps, out=W)

    if counter % 10 == 0:
        # Print logL to show progress
        print "logL[" + str(counter) + "] = ", logLs[-1]
    counter += 1


################ WRAP-UP OPERATIONS #################
# Save results to file and print out last parameter values
filename = "L" + str(dps.shape[0])
np.savez(filename, Pi=Pi, W=W,
         logLs=logLs, trueLogL=trueLogL, initW=initW)
print "results have been saved in " + filename + ".npz"
