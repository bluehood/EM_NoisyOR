#!/usr/bin/env python
# EM learning algorithm for the Noisy-OR
# author: blue, 29/03/2016
# TODO order bar matrices according to p(s_h' | y = W_dh). Delete sorting
#      routine in plotNoisyOR

import numpy as np
import signal
import argparse
import emutils as em


################ INITIAL SETUP #################
np.set_printoptions(precision=14, suppress=True, threshold=10000)

# Define option parser
parser = argparse.ArgumentParser()
parser.add_argument('-H', '--nhiddenvars', required=True, dest='nHiddenVars',
                    type=int, help='number of hidden variables')
parser.add_argument('-d', '--dpsfile', required=True, dest="dataFile",
                    help='the npy file containing the data-points')
args = parser.parse_args()

# Set problem data
# Number of hidden variables assumed present
nHiddenVars = args.nHiddenVars
# File containing the data-points/data-points
dps = np.load(args.dataFile)
# deltaDps is a quantity useful for later calculations
# deltaDps_n is 1 if dps_nd == 0 for each d, 0 otherwise
deltaDps = np.array(~np.any(dps, axis=1), dtype=int)
# Minimum value allowed for synaptic weights (max is 1-eps)
eps = 1e-2

# Create array containing all possible hidden variables configurations
hiddenVarConfs = em.genHiddenVarConfs(nHiddenVars)

# Initialise parameters
Pi = 1./nHiddenVars
W = np.random.rand(dps.shape[1], nHiddenVars)
np.clip(W, eps, 1-eps, out=W)

initW = np.copy(W) # save initial values of the parameters


################ START LEARNING #################
# From now on Ctrl-C does not interrupt execution, just sets done = True
done = [ False ] # put it in a list so the signalHandler can change the value
def sigAction(sig, frame): done[0] = True
signal.signal(signal.SIGINT, sigAction)
counter = 0;
logLs = () # The log-likelihoods evaluated at each step
for i in range(100):
    if done[0]:
        break

    # Ws are values needed both in the E and M-step, we evaluate them once
    # Ws_dhc = 1 - (W_dh * hiddenVarConfs_ch)
    Ws = 1 - np.einsum('ij,kj->ijk', W, hiddenVarConfs) 
    # prods_dc = 1 - Wbar_dc = prod{h}{1-W_dh*s_ch}
    prods = np.prod(Ws, axis=1, keepdims=True)
    # E-step: evaluate pseudo-log-joint probabilities
    pseudoLogJoints = em.pseudoLogJoint(Pi, W, hiddenVarConfs, dps, prods)

    # Evaluate new likelihood and append it to the tuple
    logLs += (em.logL(pseudoLogJoints, deltaDps, nHiddenVars, Pi),)

    # M-step
    # Pi = sum{n}{<sum{h}{hiddenVarConfs_ch}>} / (N*H)
    Pi = np.sum(em.meanPosterior(np.sum(hiddenVarConfs, axis=1),
                              pseudoLogJoints,
                              dps,
                              deltaDps)) / \
            (dps.shape[0]*nHiddenVars)

    D = hiddenVarConfs.T / (Ws*(1 - prods))
    Dtilde = np.einsum('ijk,ki->ij',
                       em.meanPosterior(D, pseudoLogJoints, dps, deltaDps),
                       dps - 1)
    C = prods*D/Ws
    Ctilde = np.sum(em.meanPosterior(C, pseudoLogJoints, dps, deltaDps),
                    axis=2)
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
         logLs=logLs, initW=initW)
print "results have been saved in " + filename + ".npz"
