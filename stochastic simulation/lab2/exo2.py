# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 14:34:41 2018

@author: toto
"""

from scipy.special import erf
import scipy
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.special as special
from scipy import stats as st
def e(x):
    return math.exp(x)


#Defines the PDF
def pdf(x):
    f= (np.sin(6*x)**2+3*np.cos(x)**2*np.sin(4*x)**2+1)*np.exp(-x**2/2)
    mass=(-4-3*e(22)-6*e(40)-3*e(54)+6*e(70)+18*e(72))*math.sqrt(math.pi*0.5)/(4*e(72))
    return f/mass;
#Computes the CDF
def cdf(x):
    f=integrate.quad(pdf, -10, x)[0]
    return f

def psi(x,C):
    return C*np.exp(-x**2/2)/(2*math.pi)

u = st.uniform.rvs(loc=0,scale=1,size=n); #uniform random distr
y = scipy.stats.norm.rvs(loc=0, scale=1, size=n)
