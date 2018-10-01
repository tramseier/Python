# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 19:48:51 2018

@author: toto
"""
import random, pylab, os
import numpy as np
from scipy import stats as st
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.stats.mstats import mquantiles
import math

def getCriticalValuesKS(n, alpha):
	#GETCRITICALVALUEKS compute the critical value for the two-sided KS test
	#statistic
	# The critical value table used below is expressed in reference to a
	# 1-sided significance level.  Need to halve the significance level for
	# a basic two-sided test.

	alpha1 = alpha / 2.

	if n <= 20:  # Small sample exact values.
		# Exact K-S test critical values are solutions of an nth order polynomial.
		# Miller's approximation is excellent for sample sizes n > 20. For n <= 20,
		# Miller tabularized the exact critical values by solving the nth order
		# polynomial. These exact values for n <= 20 are hard-coded into the matrix
		# 'exact' shown below. Rows 1:20 correspond to sample sizes n = 1:20.
		a1 = np.array([0.00500,  0.01000,  0.02500,  0.05000,  0.10000])  # 1-sided significance level

		exact = np.array([[0.99500,  0.99000,  0.97500,  0.95000,  0.90000],
        				[0.92929,  0.90000,  0.84189,  0.77639,  0.68377],
        				[0.82900,  0.78456,  0.70760,  0.63604,  0.56481],
				        [0.73424,  0.68887,  0.62394,  0.56522,  0.49265],
        				[0.66853,  0.62718,  0.56328,  0.50945,  0.44698],
        				[0.61661,  0.57741,  0.51926,  0.46799,  0.41037],
        				[0.57581,  0.53844,  0.48342,  0.43607,  0.38148],
        				[0.54179,  0.50654,  0.45427,  0.40962,  0.35831],
        				[0.51332,  0.47960,  0.43001,  0.38746,  0.33910],
	    				[0.48893,  0.45662,  0.40925,  0.36866,  0.32260],
	  			        [0.46770,  0.43670,  0.39122,  0.35242,  0.30829],
						[0.44905,  0.41918,  0.37543,  0.33815,  0.29577],
    				    [0.43247,  0.40362,  0.36143,  0.32549,  0.28470],
    				    [0.41762,  0.38970,  0.34890,  0.31417,  0.27481],
    				    [0.40420,  0.37713,  0.33760,  0.30397,  0.26588],
   					    [0.39201,  0.36571,  0.32733,  0.29472,  0.25778],
    				    [0.38086,  0.35528,  0.31796,  0.28627,  0.25039],
    				    [0.37062,  0.34569,  0.30936,  0.27851,  0.24360],
    				    [0.36117,  0.33685,  0.30143,  0.27136,  0.23735],
    				    [0.35241,  0.32866,  0.29408,  0.26473,  0.23156]])

		criticalValue  =  intrp.interpolate1d(a1 , exact[n-1,:], kind = 'cubic')(alpha1)
	else: # Large sample approximate values.
		#  alpha is a 1-sided significance level
		A = 0.09037 * (-math.log(alpha1, 10))**1.5 + 0.01515 * math.log(alpha1,10)**2 - 0.08467 * alpha1 - 0.11143
		asymptoticStat =  np.sqrt(-0.5*np.log(alpha1)/n)
		criticalValue  =  asymptoticStat - 0.16693 / n - A / n**1.5
    	
		criticalValue  =  np.min([criticalValue, 1-alpha1])
	
	return criticalValue


def LCG(X0 = 1, n = 100, a = 3, b = 0, m = 31):
	"""
	Linear Congruential Generator
	"""
	X = np.zeros(n+1)
	X[0] = X0
	for i in range(1,n+1):
		X[i] = (a * X[i-1] + b) % m

	U = X / m
	return U


def ChiSquareTest(X, K = 10, alpha = 0.1):
	K = 10
	p = np.ones(K) / K
	N = np.array([np.sum((float(i)/K < X) & (X <= (i+1.)/K)) for i in range(K)])
	QK = np.sum( (N-n*p)**2. / (n*p)) # test statistic
	critval = st.chi2.ppf(1-alpha, K-1)
	true_chi = 1 * (QK > critval)
	rej = ['cannot be', 'is']
	stat = {'Statistic': QK, 'Quantile': critval, 'Significance': alpha}
	message = 'Chi2 test: the null hypothesis H0 ' + rej[true_chi] + ' rejected at level alpha = ' + str(alpha)
	return message, stat

def KSTest(X, alpha = 0.1):
	xgrid = np.linspace(0, 1, 10001)
	Fhat = [(X<=x0).sum()/float(n) for x0 in xgrid]
	Dn = np.max(np.abs(Fhat - F(xgrid)))
	#  [Dn1, p] = st.kstest(X, F) # Comparison with the scipy.stats' builtin function
	print ('KS Test statistic: ' + str(Dn) )

	val = getCriticalValuesKS(n, alpha)
	true = 1 * (Dn > val)
	rej = ['cannot be', 'is']
	stat = {'Statistic': Dn, 'Quantile': val, 'Significance': alpha}
	message = 'KS test: the null hypothesis H0 ' + rej[true] + ' rejected at level alpha = ' + str(alpha) 
	return message, stat

def cdf(X):
	"""
	Empirical CDF
	"""
	n = X.shape[0]
	x = np.array(range(1, n+1)) / (n+1.)
	Nx = np.array([(x< x_i).sum() for x_i in x]) / float(n) # Computes empirical CDF
	xx =  np.hstack([Xs.reshape(n,1), Xs.reshape(n,1)]).flatten()
	Nxx = np.hstack([ np.hstack([Nx.reshape(n,1), Nx.reshape(n,1)]).flatten()[1:2*n], 1])
	return xx, Nxx

def GapTest(X, alpha = 0.1):
	"""
	Gap test
	"""
	aa = 0.
	bb = 0.5
	r = 5
	idx = list(set(np.where(X > aa)[0]) & set(np.where(X < bb)[0]))
	idx = np.hstack([0, np.array(idx) + 1])
	Z = idx[1:] - idx[:-1] - 1
	prob = bb - aa
	p = prob * (1 - prob) ** np.array([i for i in range(r)])
	p = np.hstack([p, (1 - prob)**r])

	Nr = np.zeros(r+1)
	for j in range(r):
		Nr[j] = np.sum( Z == j)

	nZ = len(Z)
	Nr[-1] = nZ - np.sum(Nr)
	Qr = np.sum( (Nr - p*nZ)**2. / (p*nZ))
	critval = st.chi2.ppf(1 - alpha, r)
	rej = ['cannot be', 'is']
	true = 1 * (Qr > critval)
	stat = {'Statistic': Qr, 'Quantile': critval, 'Significance' : alpha}
	message = 'Gap test: the null hypothesis H0 ' + rej[true] + ' rejected at level alpha = ' + str(alpha)
	return message, stat

def SerialTest(X, d = 2, alpha = 0.1):
	"""
	Serial statistical test 
	"""
	assert X.shape[0] % 2 == 0, 'Random sample length should be even.'
	Y = X.reshape(2, int(X.shape[0]/2)).T
	nY = Y.shape[0]
	m = 10
	K = m**d
	Nm = np.zeros(K)
	p = np.ones(K) / float(K) # True probabilities (uniform partition)
	for k in range(K):
		xl = (k % m) / float(m)
		xu = xl + 1./m
		yl = np.max([0., np.floor(k/float(m))]) / m
		yu = yl + 1./m
		Nm[k] = np.sum( ((xl< Y[:,0]) & (Y[:,0] <= xu)) * ( (yl<Y[:,1]) & (Y[:,1]<= yu)))

	rej = ['cannot be', 'is']
	Qm = np.sum( (Nm - nY*p)**2 / (nY*p))
	serval = st.chi2.ppf(1-alpha, K-1)
	true_ser = 1 * (Qm > serval)
	stat = {'Statistic' : Qm, 'Quantile': serval, 'Significance': alpha}
	message = 'Serial test: the null hypothesis H0 ' + rej[true_ser] + ' rejected at level alpha = ' + str(alpha)
	return message, stat

##########################################MAIN##################
F = lambda u: u * (u>=0) * (u<=1) + (u>1) # CDF /lambda permet de declarer une fonc en une ligne
Finv = lambda u: u * (u>=0) * (u<=1) + (u>1) # Inverse CDF

# Generate Data
n = 10000 # number of samples 
Y = st.uniform.rvs(size = (n,)) # Using Scipy's built-in function
X = LCG(n = n)[1:] # Using LCG function above 

Xs = np.sort(X) # Sorted version of data

x = np.array(range(1, n+1)) / (n+1.)
[xx, Nxx] = cdf(X)


fig = plt.figure(figsize = (8,4))
ax_qq = fig.add_subplot(121)
ax_qq.plot(x, x, 'k--', linewidth = 1.)
ax_qq.plot(Finv(x), Xs, '-.', linewidth = 1.)
ax_qq.set_title('Q-Q')
ax_cdf = fig.add_subplot(122)
ax_cdf.plot(x, F(x), 'k--', linewidth = 1.)
ax_cdf.plot(xx, Nxx, '-', linewidth = 1.)
ax_cdf.set_title('CDF')

#plt.savefig('../figures/qq_cdf_LCG_n1000.png')
plt.show()


[message_KS, stat_KS] = KSTest(X)
print (message_KS)

[message_chi2, stat_chi] = ChiSquareTest(X)
print (message_chi2)

[message_ser, stat] = SerialTest(X)
print (message_ser)

[message_gap, stat_gap] = GapTest(X)
print (message_gap)

fig1 = plt.figure(figsize = (8, 4))
ax1 = fig1.add_subplot(121)
ax1.plot(Y[:-1], Y[1:], '-', linewidth = 1.)
ax2 = fig1.add_subplot(122)
ax2.plot(X[:-1], X[1:], '-')
#plt.savefig('../figures/pairs.png')
plt.show()

