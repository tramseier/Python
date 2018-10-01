# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 10:18:56 2018

@author: toto
"""

import random, pylab, os
import numpy as np
from scipy import stats as st
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.stats.mstats import mquantiles
import math

#def cdf(X):
#	"""
#	Empirical CDF
#	"""
#	n = X.shape[0]
#	N = 1000 
#	x = np.array(range(1, N+1)) / (N+1.)
#	Nx = np.array([(x< x_i).sum() for x_i in x]) / float(n) # Computes empirical CDF
#	xx =  np.hstack([x.reshape(N,1), x.reshape(N,1)]).flatten()
#	Nxx = np.hstack([ np.hstack([Nx.reshape(N,1), Nx.reshape(N,1)]).flatten()[1:2*N], 1])
#	return xx, Nxx
def cdf(X,vec):
    n = len(vec)
    temp = np.zeros(len(X));
    for iters in range(0,len(temp)):
        temp[iters] = (((1/n)*sum(vec<=x[iters])))
    return temp

def F(x):
    if -1 <= x < 0:
        return 0
    elif 0 <= x < 2:
        return 1-(2./3)*np.exp(-x/2)
    elif 2 <= x <= 3:
        return 1
    else:
        return 0
    
def Fi(x):
    if 0 <= x < 1./3:
        return 0
    elif 1./3 <= x < 1-(2./3)*np.exp(-1):
        return -2*np.log(1.5*(-x+1))
    elif 1-(2./3)*np.exp(-1) <= x <= 1:
        return 2
    else:
        return 0
    
sz = 500
vF = np.vectorize(F,otypes=[np.float])
vFi = np.vectorize(Fi,otypes=[np.float])
x = np.linspace (-1, 3,sz)
yF = vF(x)
yFi = vFi(x)


plt.plot(x,yFi)
plt.title("CDF of invF")
plt.show()

u = st.uniform.rvs(loc=0,scale=1,size=sz); #uniform random distr
#u = np.random.normal(loc=0,scale=1,size=sz)
gr = np.linspace(0,1,sz)
X = vFi(u)
XXcdf= cdf(gr,X)
plt.plot(x,XXcdf)
plt.plot(x,yF) 
plt.title("Empir and theoretical CDF")
plt.show()

