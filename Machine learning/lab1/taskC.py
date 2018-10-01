# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 20:52:10 2018

@author: toto
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand, randn
from scipy.stats import multivariate_normal
import math #for use of pi

n, d, k = 100, 2, 2
np.random.seed(20)
X = rand(n, d)
# means = [rand(d)  for _ in range(k)]  # works for any k
means = [rand(d) * 0.5 + 0.5 , - rand(d)  * 0.5 + 0.5]  # for better plotting when k = 2
S = np.diag(rand(d))

sigmas = [S]*k # we'll use the same Sigma for all clusters for better visual results

#%%
def compute_log_p(X, mean, sigma):
    leng = np.size(X,0)
    print(leng)
#    px = np.reshape(p[:,0],(longp,1))
    for i in range(0, leng):
        mul = multivariate_normal.pdf(X, mean, sigma)
    return mul
log_ps = [compute_log_p(X, m, s) for m, s in zip(means, sigmas)]  # exercise: try to do this without looping

#%%
for m, s in zip(means, sigmas):
assignments = np.argmax(log_ps, axis=0)

colors = np.array(['red', 'green'])[assignments]
plt.scatter(X[:, 0], X[:, 1], c=colors, s=100)
plt.scatter(np.array(means)[:, 0], np.array(means)[:, 1], marker='*', s=100)
plt.show()