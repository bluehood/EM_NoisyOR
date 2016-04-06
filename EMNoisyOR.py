#!/usr/bin/env python
# EM learning algorithm for the Noisy-OR
# author: blue, 29/03/2016

import numpy as np
import signal
import argparse

np.set_printoptions(precision=10, suppress=True, threshold=50)

# Define parser
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--samplesfile', default='samples.npy',
                    dest="sFile", help='the npy file containing the samples')
parser.add_argument('-j', '--nhiddenvars', default=8, dest='nHiddenVars',
                    type=int, help='number of hidden variables')
parser.add_argument('-t', '--tparamsfile', default='t500.npz', dest="tFile",
                    help='the npz file containing the true parameters')
args = parser.parse_args()

# Set problem data
nHiddenVars = args.nHiddenVars # Number of hidden variables assumed present
samples = np.load(args.sFile)
trueParams = np.load(args.tFile)
eps = 1e-8 # minimum value allowed for synaptic weights. max is 1-eps

# Create array containing all possible hidden variables configurations
hiddenVarConfs = np.array([[0],[1]])
for i in range(nHiddenVars-1):
    hiddenVarConfs = np.vstack([ np.hstack(([x,x], [[0],[1]]))
                                 for x in hiddenVarConfs ])

# Initialise parameters to arbitrary/random values
# hiddenVarWeights = np.random.rand(nHiddenVars)
hiddenVarWeights = trueParams["hiddenVarWeights"]
# W = np.random.rand(samples.shape[1], nHiddenVars) # W[dimSample][nHiddenVars]
W = trueParams["W"]
q = np.zeros((samples.shape[1], hiddenVarConfs.shape[0]))

def jointProbs(hiddenVarWeights, W, smpls=None, hvc=None):
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
        pHiddenVarConf += [ np.prod(np.power(hiddenVarWeights, s) \
                                * np.power(1-hiddenVarWeights, 1-s),
                            axis=0) ]
    pSampleGivenS = np.array([ [
            np.prod(np.power(1-p, sample) * np.power(p, 1-sample)) 
            for p in P ]
        for sample in smpls ])
    return pSampleGivenS*np.array(pHiddenVarConf)

assert np.all(jointProbs(hvc=np.array([[0],[1]]),
        hiddenVarWeights=np.array([0.1]),
        W=np.array([[0.1],[0.5]]),
        smpls=np.array([[0,1]])) - [[ 0., 0.045 ]] <= 1e-8)


# Evaluate true log-likelihood from true parameters (for consistency checks)
trueLogL = np.sum(np.log(np.sum(jointProbs(trueParams["hiddenVarWeights"],
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
    print "Q", hiddenVarWeights

done = False
counter = 0;
logL = ()
while not done:
    pSamplesAndHidden = jointProbs(hiddenVarWeights, W)
    pSamples = np.sum(pSamplesAndHidden, axis=1)

    # Evaluate new likelihood and append it to the tuple
    logL += (np.sum(np.log(pSamples)), )

    # E-step (takes advantage of broadcasting of pSamples, hence the newaxis)
    q = pSamplesAndHidden / pSamples[:,np.newaxis]

    # M-step
    hiddenVarWeights = np.dot(np.sum(q, axis=0), hiddenVarConfs) / \
            samples.shape[0]

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

filename = "l" + str(samples.shape[0])
np.savez(filename, hiddenVarWeights=hiddenVarWeights, W=W,
         logL=logL, trueLogL=trueLogL)
print "results have been saved in " + filename + ".npz"
print "end hiddenVarWeights\n", hiddenVarWeights
print "end W\n", W
