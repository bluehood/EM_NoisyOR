import numpy as np
from math import sqrt


def genHiddenVarStates(H):
    """Generate all possible hidden variable states, i.e. all possible
configurations of the H hidden binary variables. The all-zero state is deleted
before returning the result, since it is not needed by the algorithm"""
    S = np.array([[0], [1]], dtype=int)
    for i in range(H - 1):
        S = np.vstack([np.hstack(([x, x], [[0], [1]])) for x in S])
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
    # logPriors_c = sum{h}{hvc_ch}*log(Pi_h/(1-Pi_h))
    logPriors = np.sum(S * np.log(Pi / (1 - Pi)), axis=1)
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
        / (np.sum(np.exp(plj + B), axis=0) + deltaY * np.exp(B))


def logL(plj, deltaY, H, Pi):
    """Evaluate log-likelihood logL
logL = sum{n}{log(prod{d}{delta(y_nd)} + sum{c}{exp(plj_cn)}} + \
N*sum{h}{log(1-Pi_h)}"""

    return np.sum(np.log(deltaY + np.sum(np.exp(plj), axis=0))) \
           + deltaY.size * np.sum(np.log(1 - Pi))


def genBars(D, H):
    """
    Generate data-points containing a vertical or horizontal bar.
    """
    dimMatrix = int(sqrt(D))
    nBars = min(H, 2*dimMatrix)
    # Build single-bar data-points
    bars = np.zeros((H, D))
    # "Paint" vertical bars
    for c in range(nBars / 2):
        bars[c, [i * dimMatrix + c for i in range(dimMatrix)]] = 1
    # "Paint" horizontal bars
    for c in range(nBars / 2):
        bars[ c + nBars / 2, [i + c * dimMatrix for i in range(dimMatrix)]] = 1
    return bars


def posterior(Pi, W, S, Y, deltaY = None):
    """
    Evaluate posterior probabilities p(S|Y,theta).
    Return array of posterior probabilities with shape (nHiddenConfs, N), or
    in other words (S.shape[0], Y.shape[0]).
    """
    # plj has shape (S.shape[0], Y.shape[0])
    plj = pseudoLogJoint(Pi, W, S, Y)
    # Evaluate constants B_n by which we can translate plj
    B = 200 - np.max(plj, axis = 0)
    if deltaY == None:
        # deltaY_n is 1 if Y_nd == 0 for each d, 0 otherwise (shape=(N))
        deltaY = np.array(~np.any(Y, axis=1), dtype=int)
    return np.exp(plj + B) / (np.sum(np.exp(plj + B), axis = 0) \
                              + deltaY * np.exp(B))
