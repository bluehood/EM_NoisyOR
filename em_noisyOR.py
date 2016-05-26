#!/usr/bin/env python
# Exact EM learning algorithm for the Noisy-OR generative model
# author: Enrico Guiraud, 26/05/2016

import numpy as np
import signal
import argparse
import emutils as em


################ INITIAL SETUP #################
# Define option parser
parser = argparse.ArgumentParser()
parser.add_argument('-H', '--nhiddenvars', required=True, dest='H',
                    type=int, help='number of hidden variables')
parser.add_argument('-d', '--dpsfile', required=True, dest="dataFile",
                    help='the npy file containing the data-points')
args = parser.parse_args()

# Set problem data
# Number of hidden variables assumed present
H = args.H
# Numpy array containing the data-points
Y = np.load(args.dataFile)
# Number of data-points and number of observable variables
N, D = Y.shape
# deltaY is a quantity useful for later calculations
# deltaY_n is 1 if Y_nd == 0 for each d, 0 otherwise
deltaY = np.array(~np.any(Y, axis=1), dtype=int)
# Minimum value allowed for synaptic weights (max is 1-eps)
eps = 1e-2

# Create array containing all possible hidden variable states
S = em.genHiddenVarStates(H)

# Initialise parameters
Pi = 1./H
W = np.random.rand(D, H)
np.clip(W, eps, 1-eps, out=W)
initW = np.copy(W) # save initial values of the parameters
# The log-likelihoods evaluated at each iteration step
logLs = () 

# From now on Ctrl-C will not interrupt execution but will set done[0] to True
done = [ False ]
def sigAction(sig, frame): done[0] = True
signal.signal(signal.SIGINT, sigAction)


################ START LEARNING #################
for i in range(100):
    if done[0]:
        break

    # Ws are values needed both in the E and M-step, we evaluate them once
    # Ws_dhc = 1 - (W_dh * S_ch)
    Ws = 1 - np.einsum('ij,kj->ijk', W, S) 
    # prods_dc = 1 - Wbar_dc = prod{h}{1-W_dh*s_ch}
    prods = np.prod(Ws, axis=1, keepdims=True)

    # E-step: evaluate pseudo-log-joint probabilities
    plj = em.pseudoLogJoint(Pi, W, S, Y, prods)

    # Evaluate new likelihood and append it to the tuple
    logLs += (em.logL(plj, deltaY, H, Pi),)

    # M-step
    # Pi = sum{n}{<sum{h}{S_ch}>} / (N*H)
    Pi = np.sum(em.meanPosterior(np.sum(S, axis=1), plj, Y, deltaY)) / (N*H)

    B = S.T / (Ws*(1 - prods))
    Btilde = np.einsum('ijk,ki->ij', em.meanPosterior(B, plj, Y, deltaY), Y-1)
    C = prods*B/Ws
    Ctilde = np.sum(em.meanPosterior(C, plj, Y, deltaY), axis=2)
    W = 1 + Btilde/Ctilde
    np.clip(W, eps, 1-eps, out=W)

    if i % 10 == 0:
        # Print logL to show progress
        print "logL[" + str(i) + "] = ", logLs[-1]


################ WRAP-UP OPERATIONS #################
# Save results to file
filename = "L" + str(N)
np.savez(filename, Pi=Pi, W=W, logLs=logLs, initW=initW)
print "results have been saved in " + filename + ".npz"
