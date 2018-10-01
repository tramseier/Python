# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 10:31:16 2018

@author: toto
"""

import random, pylab, os
import numpy as np
from scipy import stats as st
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.stats.mstats import mquantiles


a = np.array([25,100,10E3,10E5])
sz = 600;
A = st.uniform.rvs(loc=0,scale=1,size=sz); #uniform random distr
A.sort()
B = np.linspace(0,1,sz);#perfectly uniform distribution
xB = st.uniform.cdf(B,loc = 0,scale = 1)
xA = st.uniform.cdf(A,loc = 0,scale = 1)
xxa =  np.array([(A< x_i).sum() for x_i in A]) / float(sz)
xxb =  np.array([(B< x_i).sum() for x_i in B]) / float(sz)
xg = np.linspace(0,sz,sz)
plt.plot(xxa)
plt.plot(xxb)
plt.title('cfd a la main')
plt.show()

#plots
plt.plot(xB)
plt.plot(xA)
plt.title('CFD RNG and theory', fontsize=13)
plt.xlabel('Samples')
plt.ylabel('Probability')
plt.grid(True)
plt.show()

#kolmogorov

quant = np.linspace(0,1,10)
qA = mquantiles(xA,quant)
qB = mquantiles(xB,quant)
plt.plot(quant,qB,'--r',quant,qA,'bs')
plt.xlabel('Quantiles')
plt.ylabel('Probability')
plt.show()

#test chi2
N = 8 
count = np.zeros((N,1))
sB = np.split(B,N)

for i in range(0,N):
    count[i,0] = ((np.min(sB[i]) < A) & (A < np.max(sB[i]))).sum()
#    ssss = sum((<A<np.max(sB[i])) for j in A)

#exercise 2
m,a,b,kk = 31,3,0,100
xx = np.ones((kk,1))*20


for i in range(1,kk):
    xx[i,0] = np.remainder((a*xx[i-1,0]+b),m)
uk = xx/m


