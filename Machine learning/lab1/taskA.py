# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 14:55:37 2018

@author: toto
"""

#Task A
#%matplotlib inline
import numpy.matlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

#%load_ext autoreload
#%autoreload 2
#%%

num_samples, num_features = 10, 5
np.random.seed(10)
data = np.random.rand(num_samples, num_features)

print(data)
#substr mean and divide by std
check_m, check_std = np.mean(data, axis=0), np.std(data, axis=0) #check mean par colonne

def standardize(x):
    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)   
    return x

std_data = standardize(data)
print(std_data, "\n\n", np.mean(std_data, axis=0), "\n\n", np.std(std_data, axis=0))

#plt.plot(std_data,'ro')
#plt.title('standardized')
#plt.show()
#
#plt.plot(data,'ro')
#plt.title('non-standardized')
#plt.show()



#%%
#Task B
from scipy.spatial.distance import cdist

np.random.seed(10)
#p, q = (np.random.rand(i, 2) for i in (4, 5))
p_big, q_big = (np.random.rand(i, 80) for i in (100, 120))

#print(p, "\n\n", q)
p = np.array([[4,1],[2,2]])
q = np.array([[4,1],[2,2],[2,1]])

def naive(p, q):
    longp = np.shape(p)[0]
    longq = np.shape(q)[0]
    
    px = np.reshape(p[:,0],(longp,1))
    py = np.reshape(p[:,1],(longp,1))
    qx = np.reshape(q[:,0],(1,longq))
    qy = np.reshape(q[:,1],(1,longq))
    
    reppx = np.matlib.repmat(px,1,longq)
    reppy = np.matlib.repmat(py,1,longq)
    repqx = np.matlib.repmat(qx,longp,1)
    repqy = np.matlib.repmat(qy,longp,1)
    #tests
#    print("test test",reppx,repqx,qx,px,py,reppx,np.shape(reppx),p[:,0])
    d = np.sqrt((px - qx)**2+(py-qy)**2)
    return d
D= naive(p,q)

rows, cols = np.indices((p.shape[0], q.shape[0]))
print(rows, end='\n\n')
print(cols)

print(p[rows.ravel()], end='\n\n')
print(q[cols.ravel()])


def scipy_version(p, q):
    return cdist(p, q)

Dd = scipy_version(p,q)

#compare methods
#methods = [naive, scipy_version, tensor_broadcasting]
#timers = []
#for f in methods:
#    r = %timeit -o f(p_big, q_big)
#    timers.append(r)
#    
#plt.figure(figsize=(10,6))
#plt.bar(np.arange(len(methods)), [r.best*1000 for r in timers], log=False)  # Set log to True for logarithmic scale
#plt.xticks(np.arange(len(methods))+0.2, [f.__name__ for f in methods], rotation=30)
#plt.xlabel('Method')
#plt.ylabel('Time (ms)')
#plt.show()
#%%
#Task C

mat = np.ones((2,2))
mat2 = np.ones((10,2))
vec = np.array([[3,2],[4,4],[5,5]])
veccc = vec.T
resance = vec.T
#zer = [compute_log_p(X, m, s) for m, s in zip(means, sigmas)]  # exercise: try to do this without looping
#ress = np.dot(vec.T,mat)
x = np.array([[1,2], [3,4]])
mul = multivariate_normal.pdf(x, mean=[0, 1], cov=[[5, 2],[2,2]])
