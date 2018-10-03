# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 10:42:47 2018

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




def sort_alg(size):
    x = np.zeros((size,));
    for i in range(size):
        if i == 0:
            x[i] = np.random.uniform(0,1,1);
        else:       
            x[i] = np.random.uniform(0,1,1)**(1/i);

    for j in range(size):
        id = size-j-2;
        x[id] = x[id+1]*x[id] 
    return x
    
size = 1000;
xx = np.linspace(0,1,size);
start_sort = timeit.timeit()

u = np.random.uniform(0,1,size);
#start_sort = time.time();
u_sor=np.sort(u)
end_sort = timeit.timeit()
time_sort = end_sort-start_sort;

plt.plot(xx,sort_alg(size),label = "sort alg")
plt.plot(xx,u_sor,label = "in-built sort");
plt.legend()
plt.show()

x = np.zeros((size,));
time = np.zeros((size,1));
start_alg = timeit.timeit();

end_alg = timeit.timeit()

time_alg = end_alg -start_alg

#simplex
dia = np.ones(2,);
a = -np.diag(dia,-1)+ np.eye(3,3);

#plot simplex
xs = np.zeros((size,1));
ys = np.zeros((size,1));
zs = np.zeros((size,1));

for i in range(0,size):
    point = np.dot(a,sort_alg(3));
    xs[i] = point[0]; 
    ys[i] = point[1]; 
    zs[i] = point[2]; 

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#Defines the values. CHANGE THESE LINES
#x =[1,2,3,4,5,6,7,8,9,10]
#y =[5,6,2,3,13,4,1,2,4,8]
#z =[2,3,3,3,5,7,9,11,9,10]
ax.scatter(xs, ys, zs, c='r', marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()