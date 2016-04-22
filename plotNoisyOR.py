#!/usr/bin/env python
# Plot log-likelihood and compare true parameters with learned parameters
# author: blue, 29/03/2016

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import argparse
from sys import exit

# Define parser
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--parFiles', dest='parFile',
                    help="""The body of the names of the files containing the
parameters. The samples file will be set to
'sPARFILE.npy'. The true parameters file will be set
to 'tPARFILE.npz'. The learned parameters file will be set to 
'lPARFILE.npz'. This is a commodity option to save
typing. This option is overridden by the -s, -t and -l options
if they are present.""")
parser.add_argument('-t', '--truth', dest="truep",
                    help='the npz file containing the true parameters')
parser.add_argument('-l', '--learned', dest="learnedp",
                    help='the npz file containing the learned parameters')
parser.add_argument('-s', '--samples', dest="samples",
                    help='the npy file containing the samples')
args = parser.parse_args()
if not args.parFile:
    if not (args.samples and args.truep and args.learnedp):
        print """Please provide samples, true and learned parameters filenames.
The -p option can be used as a shorthand if filenames have the same body.
Please use the -h option for more information"""
        exit(1)

def clear_axes(plot):
    axes = plot.get_axes()
    axes.set_xticklabels([])
    axes.set_xticks([])
    axes.set_yticklabels([])
    axes.set_yticks([])


def compare(A, B):
    """Ad-hoc matrix comparison that uses a metric useful to
    list bars matrices in an ordered fashion"""
    Ascore = np.sum(np.rint(A)*np.exp2(np.arange(A.size)).reshape(A.shape))
    Bscore = np.sum(np.rint(B)*np.exp2(np.arange(B.size)).reshape(B.shape))
    return cmp(Ascore,Bscore)

# Load true parameters
tp = np.load(args.truep or ('t' + args.parFile + '.npz'))
# Load log-likelihood and learned parameters
lp = np.load(args.learnedp or ('l' + args.parFile + '.npz'))
# Samples
samples = np.load(args.samples or ('s' + args.parFile + '.npy'))
# Plot log-likelihood and true log-likelihood
plt.figure()
plt.plot(range(lp["logLs"].size),
         lp["logLs"],
         range(lp["logLs"].size),
         lp["trueLogL"].repeat(lp["logLs"].size))
plt.title("True vs learned log-likelihood")
plt.xlabel("iterations")
plt.ylabel("log-likelihood")
# Plot the first twelve samples
dimMatrix = sqrt(samples.shape[1])
samples = samples[:12]

plt.figure()
for nPlot in range(12):
    plt.subplot(4, 3, nPlot+1)
    plot = plt.imshow(samples[nPlot].reshape(dimMatrix, dimMatrix),
                      cmap="Greys", interpolation='none')
    clear_axes(plot)
    if nPlot == 1:
            plt.title('First 12 samples')

# Build grid of heat-maps to compare true and learned parameters
nMatrices = max(tp["W"].shape[1], lp["W"].shape[1])
# matrices[0] is the initial matrices, matrices[1] is the learned ones,
# matrices[2] the ground-truth matrices
matrices = [ np.transpose(lp["initW"]).reshape(lp["W"].shape[1],
                                               dimMatrix,
                                               dimMatrix), 
             np.transpose(lp["W"]).reshape(lp["W"].shape[1],
                                           dimMatrix,
                                           dimMatrix),
             np.transpose(tp["W"]).reshape(tp["W"].shape[1],
                                           dimMatrix,
                                           dimMatrix) ]
matrices[1] = np.array(sorted(matrices[1], cmp=compare))
matrices[2] = np.array(sorted(matrices[2], cmp=compare))
titles = ("Initial weights", "Learned weights", "True weights")
plt.figure(figsize=(plt.rcParams['figure.figsize'][0],
                    plt.rcParams['figure.figsize'][1]*2))
for nMat in range(nMatrices):
    for i in range(3):
        if matrices[i].shape[0] > nMat:
            plt.subplot(nMatrices, 3, nMat*3+i+1)
            plot = plt.imshow(matrices[i][nMat], cmap="Greys", aspect='equal',
                              vmin=0, vmax=1, interpolation='none')
            clear_axes(plot)
            if nMat == 0:
                plt.title(titles[i])
        
plt.figure()
plt.subplot(2,1,1)
plot = plt.imshow(np.atleast_2d(lp["Pi"]), interpolation='none',
                  vmin=0., vmax=.5)
clear_axes(plot)
plt.title("Subpopulations' probabilities (learned over ground-truth)")
plt.subplot(2,1,2)
plot = plt.imshow(np.atleast_2d(tp["Pi"]), interpolation='none',
                  vmin=0., vmax=.5)
clear_axes(plot)
# FIXME plt.colorbar()

plt.show()
