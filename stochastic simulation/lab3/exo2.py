# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 14:31:02 2018

@author: toto
"""

from scipy.special import erf
import scipy
import time
import timeit
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.special as special
from scipy import stats as st
def e(x):
    return math.exp(x)
from mpl_toolkits.mplot3d import Axes3D#for scatter plot

def simple_sample_gauss(mu,sig,size):
        n = np.shape(mu)[0];
        samples = np.empty((2,1), int);
        l = np.linalg.cholesky(sig);

        for i in range(0,size):   
            y = np.random.normal(0,1,n);
            temp = l@y;
            temp = np.reshape(temp,(n,1));
            samples = np.hstack((samples,mu +temp ));
        return samples

            
            
            
def sample_gauss(mu,sig,size):
    n = np.shape(mu)[0];
#    samples = np.zeros((n,size))
    samples = np.empty((2,1), int)
    for i in range(0,size):   
        isig = np.linalg.inv(sig);
#        verif = isig@sig;
        l = np.linalg.cholesky(isig);
#        chap = l@l.T;
        z = np.random.normal(0,1,n);
        y = np.dot(np.linalg.inv(l.T),z);
        y = np.reshape(y,(n,1));
#        print(y.shape)
        samples = np.hstack((samples,mu + y));
    return samples

nb_samples = 10**3;
sig = np.array([[1,2],[2,5]]);
mu = np.array([[2],[1]]);
res = sample_gauss(mu,sig,nb_samples);
res_simple = simple_sample_gauss(mu,sig,nb_samples)

#plotting histograms from both methods
xedges = [0, 1, 3, 5,6];
yedges = [0, 2, 3, 4, 6];

H, xedges, yedges = np.histogram2d(res[0,:],res[1,:], bins=(xedges, yedges))
H = H.T  # Let each row list bins with common y range.

fig = plt.figure(figsize=(7, 3))
ax = fig.add_subplot(131, title='imshow: square bins')
plt.imshow(H, interpolation='nearest', origin='low',
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

H2, xedges, yedges = np.histogram2d(res_simple[0,:],res_simple[1,:], bins=(xedges, yedges))
H2 = H2.T  # Let each row list bins with common y range.

fig = plt.figure(figsize=(7, 3))
ax = fig.add_subplot(131, title='imshow: square bins')
plt.imshow(H2, interpolation='nearest', origin='low',
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])