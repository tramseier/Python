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
    return C*np.exp(-x**2/2)/(np.sqrt(2*math.pi))

def cdff(x,vec):
    n = len(vec)
    temp = np.zeros(len(x));
    for iters in range(0,len(temp)):
        temp[iters] = (((1/n)*sum(vec<=x[iters])))
    return temp

#Efficient ratio immediately 
def eff_ratio(x):
     return (np.sin(6*x)**2+3*np.cos(x)**2*np.sin(4*x)**2+1)/(easy_C)

fhat = lambda x:(np.sin(6*x)**2+3*np.cos(x)**2*np.sin(4*x)**2+1)*np.exp((-x**2)/2.)
c = 11;
x = np.linspace(-4,4,100);
pd = fhat(x);
plt.plot(x,pd,label = "fhat") #choose the right c value
plt.plot(x,psi(x,c),label = "normal")
plt.legend()
plt.show()

#second exercise

n = 1000;
#AR method
M,smooth = 10,100
grid = np.linspace(-M,M,smooth)
accept = np.zeros(int(n));
acc_count = 0
iteration = 0

while acc_count < n:
    y = np.random.standard_normal((1,))
    u = np.random.uniform(0,1,1)
    if (u <= fhat(y/(psi(y,c)))):
        accept[acc_count] = y;
        acc_count += 1;
    iteration += 1;

buffer2 = np.zeros(smooth)
for iters in range(0,smooth):
    buffer2[iters] = cdf(grid[iters])

plt.plot(grid,buffer2)
plt.plot(grid,cdff(grid,accept))
plt.title("A/R method")
plt.show()

#comparison of k
exact_k = 0.1696542774;
prob_acc = n/ iteration;
exp_k = 1/(prob_acc*c)