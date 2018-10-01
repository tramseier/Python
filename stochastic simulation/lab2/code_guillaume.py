# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 10:17:44 2018

@author: Guillaume
"""

from scipy.special import erf
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.special as special

# /!\ extras /!\ 
def e(x):
    return math.exp(x)

#Computes the empirical CDF
# x: abscisse, vec: values we wish to analyze

def cum_emp(x,vec):
    n = len(vec);
    temp = np.zeros(len(x));
    for iters in range(0,len(temp)):
        temp[iters] = (((1/n)*sum(vec<=x[iters])));
    return(temp)

# @exercice 1 
    
def F(x):
    if((x>=-1) &  (x<0)):
        return 0
    elif((x>=0) & (x<2)):
        return 1-(2/3)*math.exp(-x/2)
    elif((x>=2) & (x<=3)):
        return 1
    else:
        return 0
        
smooth = 200
a = -1
b = 3
epsilon = 0.000001
    
vec = np.linspace(a,b,smooth)

buffer = np.zeros(smooth)

for iters in range(0,smooth):
    buffer[iters] = F(vec[iters])

    
''' @requires u between 0 and 1 
=> no vectorization for now... actually one has really to work elementwise
'''

def F_inv(u):
    if(u==0): # not probable... :p 
        return -1
    elif(u<F(0)):
        return 0
    elif((u>=F(0)) and (u<F(2-epsilon))):
        return -2*math.log((3/2)*(1-u))
    else:
        return 2
   
''' #tests
vec_bis_test = np.linspace(0,1,smooth)
buffer_bis_test = np.zeros(smooth)

for iters in range(0,smooth):
    buffer_bis_test[iters] = F_inv(vec_bis_test[iters])

plt.plot(vec_bis_test,buffer_bis_test)
plt.show()
'''

n = 200  
 
buffer_bis = np.zeros(n)
unif_bis = np.random.uniform(0,1,n)


for iters in range(0,n):
    buffer_bis[iters] = F_inv((unif_bis[iters]))

plt.plot(vec,buffer)
plt.plot(vec,cum_emp(vec,buffer_bis))
plt.show()


# @exercice 2

#Defines the known weight function
def f2_tilde(x):
    return (np.sin(6*x)**2+3*np.cos(x)**2*np.sin(4*x)**2+1)*np.exp(-x**2/2)


def phi_standard(x):#normal pdf
    return e(-x**2 / 2) * (1/math.sqrt(2*math.pi))

#Defines the true PDF
def f2(x):
    f= (np.sin(6*x)**2+3*np.cos(x)**2*np.sin(4*x)**2+1)*np.exp(-x**2/2)
    mass=(-4-3*e(22)-6*e(40)-3*e(54)+6*e(70)+18*e(72))*math.sqrt(math.pi*0.5)/(4*e(72))
    return f/mass;

#Computes the true CDF
def F2(x):
    f=integrate.quad(f2, -10, x)[0]
    return f

easy_C = 5*math.sqrt(2*math.pi)

#Efficient ratio immediately 
def eff_ratio(x):
     return (np.sin(6*x)**2+3*np.cos(x)**2*np.sin(4*x)**2+1)/(easy_C)
n2 = 500

acceptance_counter = 0
iterations = 0

buffer2_bis = np.zeros(int(n2))


''' regular A-R algorithm without any improvement '''
while acceptance_counter<n2:
    
     y = np.random.standard_normal((1,))
     u = np.random.uniform(0,1,1)
     
     if(u<=eff_ratio(y)):
        buffer2_bis[acceptance_counter] = y
        acceptance_counter+=1
        
     iterations+=1  

empirical_acceptance_ratio = n2/iterations 
print(empirical_acceptance_ratio)

# as it should be (1/KC) we can estimate it (poorly i think) by 

K_estimate = 1/(empirical_acceptance_ratio*easy_C)
print(K_estimate)
mass=(-4-3*e(22)-6*e(40)-3*e(54)+6*e(70)+18*e(72))*math.sqrt(math.pi*0.5)/(4*e(72))
true_K  = (1/mass)
print(true_K)

M = 1e1
smooth2 = 100

vec2 = np.linspace(-M,M,smooth2)

''' it will take a while :p '''
buffer2 = np.zeros(smooth2)

for iters in range(0,smooth2):
    buffer2[iters] = F2(vec2[iters]) #create true cdf vector for plotting

plt.plot(vec2,buffer2) #plottting true cdf
plt.plot(vec2,cum_emp(vec2,buffer2_bis))
plt.show()

# @exercice 3 







