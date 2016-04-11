#!/usr/bin/env python
# EM learning algorithm for the Noisy-OR
# author: blue, 29/03/2016

import numpy as np
import signal
import argparse

np.set_printoptions(precision=14, suppress=True, threshold=50)

# this seed causes problems with the data in results/j6d9n500/wrong2
#np.random.seed(17564234)

# Define parser
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--samplesfile', required=True, dest="sFile",
                    help='the npy file containing the samples')
parser.add_argument('-j', '--nhiddenvars', required=True, dest='nHiddenVars',
                    type=int, help='number of hidden variables')
parser.add_argument('-t', '--tparamsfile', required=True, dest="tFile",
                    help='the npz file containing the true parameters')
args = parser.parse_args()

# Set problem data
nHiddenVars = args.nHiddenVars # Number of hidden variables assumed present
samples = np.load(args.sFile)
trueParams = np.load(args.tFile)
eps = 1e-8 # minimum value allowed for synaptic weights. max is 1-eps

# Create array containing all possible hidden variables configurations
hiddenVarConfs = np.array([[0],[1]], dtype=int)
for i in range(nHiddenVars-1):
    hiddenVarConfs = np.vstack([ np.hstack(([x,x], [[0],[1]]))
                                 for x in hiddenVarConfs ])

# Initialise parameters to random values
Pi = np.random.rand()
W = np.random.rand(samples.shape[1], nHiddenVars) # W[dimSample][nHiddenVars]

# Initialise parameters to the ground-truth values
# Pi = trueParams["Pi"]
# W = trueParams["W"]

q = np.zeros((samples.shape[1], hiddenVarConfs.shape[0]))
initW = W

def jointProbs(Pi, W, smpls=None, hvc=None):
    """Takes the parameters and returns a matrix p[sample][hiddenVarConfs].
    Each element of the matrix is the joint probablity \
                    p(sample, hiddenVarConf)"""

    if smpls is None:
            smpls = samples
    if hvc is None:
            hvc = hiddenVarConfs

    P = []
    pHiddenVarConf = []
    #TODO do everything via dot, tensordot ecc
    for s in hvc:
        P += [ np.prod(1-W*s, axis=1) ];
        pHiddenVarConf += [ np.prod(np.power(Pi, s) \
                                * np.power(1-Pi, 1-s),
                            axis=0) ]
    pSampleGivenS = np.array([ [
            np.prod(np.power(1-p, sample) * np.power(p, 1-sample)) 
            for p in P ]
        for sample in smpls ])
    return pSampleGivenS*np.array(pHiddenVarConf)

assert np.all(jointProbs(hvc=np.array([[0],[1]]),
        Pi=np.array([0.1]),
        W=np.array([[0.1],[0.5]]),
        smpls=np.array([[0,1]])) - [[ 0., 0.045 ]] <= 1e-8)


# Evaluate true log-likelihood from true parameters (for consistency checks)
trueLogL = np.sum(np.log(np.sum(jointProbs(trueParams["Pi"],
                                            trueParams["W"]),
                                            axis=1)))

def signalHandler(signal, frame):
    global done
    print "Quitting on keyboard interrupt!"
    done = True
signal.signal(signal.SIGINT, signalHandler)
    
def debugPrint():
    print "q", q
    print "W", W
    print "Pi", Pi

done = False
counter = 0;
logL = ()
while not done:
    pSamplesAndHidden = jointProbs(Pi, W)
    pSamples = np.sum(pSamplesAndHidden, axis=1)

    # Evaluate new likelihood and append it to the tuple
    logL += (np.sum(np.log(pSamples)), )

    # E-step (np.newaxis is needed to use broadcasting)
    q = pSamplesAndHidden / pSamples[:,np.newaxis]

    # M-step
    Pi = np.dot(np.sum(q, axis=0), np.sum(hiddenVarConfs, axis=1)) / \
            (samples.shape[0]*nHiddenVars)

    # Wtilde has shape (dimSample, nHiddenVars, nHiddenVarsConfs) and
    # Wtilde[d,j,c] = Prod_{j'!=j}{1 - W[d,j']*hiddenVarConfs[c,j']}
    tmp = 1 - np.einsum('ij,kj->ijk',
                        W, hiddenVarConfs) # faster than * plus np.newaxis
    Wtilde = np.stack([ np.prod(np.delete(tmp, j, 1), axis=1) \
            for j in range(tmp.shape[1]) ], axis=1)
    
    denominators = 1 - Wtilde*tmp # faster than np.prod(tmp, axis=1)
    denominators = (1 - denominators)*denominators
    denominators[:,:,0] = 1 # Hack to resolve 0/0 operations to 0
    D = np.einsum('ijk,kj->ijk', Wtilde, hiddenVarConfs) / denominators
    C = Wtilde*D
    sum_nCavg = np.dot(C, np.sum(q, axis=0))
    sum_nD_y = np.einsum('ijk,ki->ij', np.tensordot(D, q, (2,1)), samples - 1)
    W = 1 + sum_nD_y/sum_nCavg
    W[W<eps] = eps
    W[W>1-eps] = 1-eps

    # Print logL to show progress
    counter += 1
    if counter % 10 == 0:
        debugPrint()
        print "logL[" + str(counter) + "] = ", logL[-1]

# Evaluate errors
smallval = np.min(trueParams["W"])
bigval = np.max(trueParams["W"])
smalldiff = np.abs(W[W<0.5] - smallval)
bigdiff = np.abs(W[W>=0.5] - bigval) 
Werror = np.max(np.append(smalldiff, bigdiff))

filename = "l" + str(samples.shape[0])
np.savez(filename, Pi=Pi, W=W,
         logL=logL, trueLogL=trueLogL, initW=initW)
print "end Pi\n", Pi
print "end W (max error: " + str(Werror) + ")\n", W
print "results have been saved in " + filename + ".npz"
