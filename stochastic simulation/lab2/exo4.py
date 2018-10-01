# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 20:02:12 2018

@author: toto
"""

from scipy.special import erf
import scipy
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.special as special
from scipy import stats as st

def psi(x):
    return np.exp(-x**2/2)/(np.sqrt(2*math.pi))

def kk(x):
    return np.exp(-x**2/2)/(np.sqrt(2*math.pi))

def k_sig(x,band):
    return (1/band)*kk(x/band)

def ff(x,samples,band):
    n = len(samples);
    nx = len(x);
    res = np.zeros(nx,);
    for i in range(0,nx):
        for j in range(0,n):
            res[i] += (k_sig(x[i]-samples[j],band))/n;
    return res

def sig(samples):
    n = len(samples);
    var = np.std(samples);
    return var*n**(-1/5.)

c, k = 2,4;
M = 10;
band,size = 1,100;

sizes = [100, 500];
for sz in sizes:
    xx = scipy.stats.burr12.rvs(c, k,0, 1, sz);
    x = np.linspace(-M,M,sz);
    band = sig(xx);
    plt.plot(x,ff(x,xx,band),label = sz)    
    
rv = scipy.stats.burr12(c, k);
x = np.linspace(-M,M,size);
plt.plot(x, rv.pdf(x), 'k-', lw=2, label='Theoretical Blurr')
plt.legend()
plt.show()