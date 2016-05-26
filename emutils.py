#!/usr/bin/env python
import numpy as np

def genHiddenVarStates(H):
    """Generate all possible hidden variable states, i.e. all possible
configurations of the H hidden binary variables. The all-zero state is deleted
before returning the result, since it is not needed by the algorithm"""
    S = np.array([[0],[1]], dtype=int)
    for i in range(H-1):
        S = np.vstack([ np.hstack(([x,x], [[0],[1]])) for x in S ])
    # Remove the all-zero configuration:
    # it is not required in the EM-algorithm and causes 0/0 calculations
    S = np.delete(S, 0, 0)
    return S


def pseudoLogJoint(Pi, W, S, Y, prods=None):
    """Evaluate pseudo-log-joints K*log(p(S[c], Y[n])) where K is a constant
that does not depend on the state and the data-point considered, and p(s,y) is
the joint probability of a certain hidden state s and an observable state y."""
    if prods is None:
        # prods_dc = 1 - Wbar_dc = prod{h}{1-W_dh*s_ch}
        prods = np.prod(1 - np.einsum('ij,kj->ijk', W, S), axis=1)
    else:
        # make sure the prods passed have the appropriate shape (DxC)
        prods = prods.reshape(W.shape[0], S.shape[0])

    # logPy_nc = sum{d}{y_nd*log(1/prods_dc - 1) + log(prods_dc)}
    logPy = np.dot(Y, np.log(1/prods - 1)) + np.sum(np.log(prods), axis=0)
    # logPriors_c = sum{h}{hvc_ch}*log(Pi/(1-Pi))
    logPriors = np.sum(S, axis=1)*np.log(Pi/(1-Pi))
    # return plj_cn
    return np.transpose(logPriors + logPy)


def meanPosterior(g, plj, Y, deltaY):
    """Evaluate the mean over the posterior probability distribution of the
(possibly multidimensional) array g.
g is assumed to have the axis over which the mean is to be performed as last.

An array with the same number of dimensions of g is returned, but now
the last axis represents the mean of g relative to each different data-point.
    
The calculation performed is equivalent to np.dot(g, q) where q_cn are the 
posterior probabilities of each hidden variable configuration c given the 
data-point y_n"""

    # Evaluate constants B_n by which we can translate plj
    B = 200 - np.max(plj, axis=0)

    # sum{c}{g_ic*exp(plj_cn + B)} /
    #   (sum{c}{exp(plj_cn + B)} + prod{d}{delta(y_nd)*exp(B))
    return np.dot(g, np.exp(plj + B)) \
           / (np.sum(np.exp(plj + B), axis=0) + deltaY*np.exp(B))


def logL(plj, deltaY, H, Pi):
    """Evaluate log-likelihood logL
logL = sum{n}{log(prod{d}{delta(y_nd)} + sum{c}{exp(plj_cn)}} + N*H*log(1-Pi)"""

    return np.sum(np.log(deltaY + np.sum(np.exp(plj), axis=0))) \
           + deltaY.size*H*np.log(1-Pi)
