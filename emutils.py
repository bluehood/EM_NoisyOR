#!/usr/bin/env python
import numpy as np

def genHiddenVarConfs(H):
    hiddenVarConfs = np.array([[0],[1]], dtype=int)
    for i in range(H-1):
        hiddenVarConfs = np.vstack([ np.hstack(([x,x], [[0],[1]]))
                                     for x in hiddenVarConfs ])
    # Remove the all-zero configuration:
    # it is not required in the EM-algorithm and causes 0/0 calculations
    hiddenVarConfs = np.delete(hiddenVarConfs, 0, 0)
    return hiddenVarConfs

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


def pseudoLogJoint(Pi, W, hiddenVarConfs, dps, Ws=None):
    """Takes the parameters and returns a matrix M[hiddenVarConfs][dp].
Each element of the matrix is the pseudo-log-joint probablity \
B*log(p(hiddenVarConf, dp))"""
    if Ws is None:
        Ws = 1 - np.einsum('ij,kj->ijk', W, hiddenVarConfs)

    # prods_dc = 1 - Wbar_dc = prod{h}{1-W_dh*s_ch}
    prods = np.prod(Ws, axis=1)
    # logPy_nc = sum{d}{y_nd*log(1/prods_dc - 1) + log(prods_dc)}
    logPy = np.dot(dps, np.log(1/prods - 1)) + \
                np.sum(np.log(prods), axis=0)
    # logPriors_c = sum{h}{hvc_ch}*log(Pi/(1-Pi))
    logPriors = np.sum(hiddenVarConfs, axis=1)*np.log(Pi/(1-Pi))
    # return pseudoLogJoints_cn
    return np.transpose(logPriors + logPy)


def meanPosterior(g, pseudoLogJoints, dps, deltaDps):
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
