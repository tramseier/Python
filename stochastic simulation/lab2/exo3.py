# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 18:01:25 2018

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
def e(x):
    return math.exp(x)

def psi(x):
    return np.exp(-x**2/2)/(2*math.pi)

def cdff(x,vec): # x: plotting vec, vec: actual numbers we sampled
    n = len(vec)
    temp = np.zeros(len(x));
    for iters in range(0,len(temp)):
        temp[iters] = (((1/n)*sum(vec<=x[iters])))
    return temp

def theta(x):
    return 2*math.pi*x

def rho(x):
    return np.sqrt(-2*np.log(x))

def g(x):
    return np.exp(-np.abs(x))/2;

def sample_g(n): #n = nb of samples
    res = np.zeros(n);
    for i in range(0,n):
        bi = np.random.binomial(1,0.5);
        y = np.random.exponential(0.5); #lambda = 1/scale 
        if bi == 0:
            res[i] = -y;
        elif bi ==1:
            res[i] = y;
    return res
          
n = 500;
  
#finding c, think I should finite analytically...
c = 2.5; #looks good
samples = sample_g(n);
bins = np.linspace(-5, 5, 40);
bin_centers = 0.5*(bins[1:] + bins[:-1])
pdf = c*st.norm.pdf(bin_centers)
histogram, bins = np.histogram(samples, bins=bins, normed=True)
plt.plot(bin_centers, histogram, label="Histogram of samples")
plt.plot(bin_centers, pdf, label="PDF of perfect normal")
plt.legend()
plt.show()

#box muller
start_box = time.time()

xx,yy = np.zeros(n),np.zeros(n); 
for i in range(0,n):
    u,v = np.random.uniform(0,1,1),np.random.uniform(0,1,1);
    r = rho(u);
    tet = theta(v);
     
    xx[i] = r*np.cos(tet);
    yy[i] = r*np.sin(tet);
end_box = time.time()
time_box = end_box-start_box;

#AR algo
start_ar = time.time()
accept = np.zeros(int(n));
acc_count = 0
iteration = 0

while acc_count < n:
    y = sample_g(1);
    u = np.random.uniform(0,1,1)
    if (u <= psi(y)/(c*g(y))):
        accept[acc_count] = y;
        acc_count += 1;
    iteration += 1;
end_ar = time.time()
time_ar = end_ar - start_ar;

################tests
#test sample g
#s = sample_g(n);
#plt.plot(grid,cdff(grid,s));

#test muller
M,smooth = 10, 200;
grid = np.linspace(-M,M,smooth);
normal = np.random.standard_normal((1000,));

plt.plot(grid,cdff(grid,yy),label = "Box muller")
plt.plot(grid,cdff(grid,normal),label = "Perfect distr")
plt.legend()
plt.show()

#test AR
plt.plot(grid,cdff(grid,normal),label = "Perfect distr")
plt.plot(grid,cdff(grid,accept),label = "AR pdf")
plt.legend()
plt.show()