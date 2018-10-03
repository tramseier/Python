# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 16:04:34 2018

@author: toto
"""

from scipy.special import erf
import scipy
import timeit
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.special as special
from scipy import stats as st


def simple_sample_gauss(mu,sig,nb_samples):
    n = np.shape(mu)[0];
    samples = np.empty((n,1), int);
    l = np.linalg.cholesky(sig);

    for i in range(0,nb_samples):   
        y = np.random.normal(0,1,n);
        temp = l@y;
        temp = np.reshape(temp,(n,1));
        samples = np.hstack((samples,mu +temp));
    return samples

def mu(t):
    n = len(t);
    res = np.zeros(n,);
    for i in range(0,n):
        res[i] = math.sin(2*math.pi*t[i])
    res = np.reshape(res,(n,1));
    return res

def sig(t,ro):#t is a the time vector
    n = len(t);
    sig = np.zeros((n,n));
    for i in range(0,n):
        for j in range(0,n):
            sig[i,j] = math.exp(-np.abs(i-j)/ro);        
    return sig

def brownian(t,ro):# t = vector of time
    muu = mu(t);
    sigg = sig(t,ro);
    n = len(t);
    nb_samples = 1;
    return simple_sample_gauss(muu,sigg,nb_samples)


time_length = 100;
t = np.linspace(0,time_length,time_length);
for ro in [5/10,1/100,8/100]:   
    y = brownian(t,ro);
    plt.plot(t,y[:,1],label = ro)
plt.legend()
plt.show()